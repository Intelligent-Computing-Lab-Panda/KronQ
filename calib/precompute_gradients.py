# --- repo path bootstrap: make common/, runtime/, calib/ importable from any entry point ---
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not all(_os.path.isdir(_os.path.join(_d, s)) for s in ('common', 'runtime', 'calib')):
    _d = _os.path.dirname(_d)
for _s in ('common', 'runtime', 'calib'):
    _p = _os.path.join(_d, _s)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import os
import argparse
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

import rotation_utils
import hadamard_utils
import quant_utils


def precompute_fisher_gradients(model, dataloader, dev, args, save_dir):
    """Compute G using Fisher sampling (like LLM-Surgeon)."""
    os.makedirs(save_dir, exist_ok=True)

    model.train()

    if torch.is_inference_mode_enabled():
        raise RuntimeError("Cannot compute gradients in inference mode")

    layers = model.model.layers
    nlayers = len(layers)

    sublayer_names = [
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
    ]

    # Qwen-family QK-Norm: q_proj/k_proj outputs are reshaped and RMSNorm'd, so a
    # module backward hook is unreliable there — hook the output tensor directly.
    is_qk_norm = hasattr(layers[0].self_attn, 'q_norm')
    qk_norm_layers = {'self_attn.q_proj', 'self_attn.k_proj'} if is_qk_norm else set()
    if is_qk_norm:
        print("QK-Norm detected: using forward+tensor hook for q_proj/k_proj")

    # Enable gradients for projection layers only
    for n, p in model.named_parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if 'proj' in n:
            p.data = p.data.detach().clone()
            p.requires_grad_(True)

    model = model.to(dev)

    fisher_samples = args.fisher_samples

    for layer_idx in range(nlayers):
        print(f"\n{'=' * 50}")
        print(f"Processing Layer {layer_idx}/{nlayers - 1}")
        print(f"{'=' * 50}")

        layer_dir = os.path.join(save_dir, f"layer_{layer_idx}")

        if os.path.exists(layer_dir) and len(os.listdir(layer_dir)) == len(sublayer_names):
            print(f"Layer {layer_idx} already computed, skipping...")
            continue

        os.makedirs(layer_dir, exist_ok=True)

        # Storage for G matrices (accumulate on CPU)
        G_accum = {name: None for name in sublayer_names}
        G_count = {name: 0 for name in sublayer_names}

        # Temporary storage for current backward pass
        current_grads = {}
        handles = []
        tensor_handles = []  # tensor-level hooks for QK-Norm q/k; removed after the layer

        def make_grad_hook(name):
            def hook(module, grad_input, grad_output):
                g = grad_output[0].detach()
                g = g.reshape(-1, g.shape[-1]).float()  # [batch*seq, out_features]
                current_grads[name] = g

            return hook

        # QK-Norm q/k: capture d(loss)/d(projection output) via forward hook +
        # retain_grad + tensor hook, bypassing the reshape/RMSNorm that follows.
        def make_qknorm_hook(name):
            def fwd_hook(module, inp, output):
                if not output.requires_grad:
                    return
                output.retain_grad()

                def tensor_hook(grad):
                    current_grads[name] = grad.detach().reshape(-1, grad.shape[-1]).float()

                tensor_handles.append(output.register_hook(tensor_hook))

            return fwd_hook

        # Register hooks
        layer = layers[layer_idx]
        for name in sublayer_names:
            try:
                module = layer.get_submodule(name)
                if hasattr(module, 'module'):
                    module = module.module
                if name in qk_norm_layers:
                    handles.append(module.register_forward_hook(make_qknorm_hook(name)))
                else:
                    handles.append(module.register_full_backward_hook(make_grad_hook(name)))
            except AttributeError:
                print(f"Warning: {name} not found in layer {layer_idx}")

        # Forward + backward with FISHER SAMPLING
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Layer {layer_idx} gradients")):
            input_ids = batch[0].to(dev)

            with torch.enable_grad():
                outputs = model(input_ids)
                logits = outputs.logits

                shift_logits = logits[:, :-1, :].contiguous()

                # === FISHER SAMPLING: Sample from model's distribution ===
                with torch.no_grad():
                    probs = F.softmax(shift_logits, dim=-1)

                for fs in range(fisher_samples):
                    current_grads.clear()

                    # Sample labels from model's output distribution
                    sampled_labels = torch.multinomial(
                        probs.view(-1, probs.size(-1)),
                        num_samples=1
                    ).squeeze(-1)

                    # Compute loss with SAMPLED labels
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        sampled_labels,
                        reduction='mean'
                    )

                    # Backward
                    model.zero_grad()
                    loss.backward(retain_graph=(fs < fisher_samples - 1))

                    # Accumulate G = g.T @ g on CPU
                    for name, g in current_grads.items():
                        g = g.cpu()
                        n = g.shape[0]

                        g_normalized = g * math.sqrt(2.0 / n)
                        G_batch = g_normalized.T @ g_normalized

                        if G_accum[name] is None:
                            G_accum[name] = G_batch
                        else:
                            G_accum[name] += G_batch
                        G_count[name] += 1

                model.zero_grad()

        for h in handles:
            h.remove()
        for h in tensor_handles:
            h.remove()

        # Average and save G
        for name in sublayer_names:
            if G_accum[name] is None:
                continue

            G = G_accum[name] / G_count[name]

            fname = name.replace('.', '_') + '_G.pt'
            save_path = os.path.join(layer_dir, fname)
            torch.save(G, save_path)
            print(f"  Saved {name}: {G.shape}, count={G_count[name]}")

        del G_accum, current_grads
        gc.collect()
        torch.cuda.empty_cache()

    # Save metadata
    metadata = {
        'nsamples': args.nsamples,
        'fisher_samples': args.fisher_samples,
        'nlayers': nlayers,
        'sublayer_names': sublayer_names,
        'model_name': args.model,
        'format': 'G_matrix_fisher',
        'rotated': args.rotate,
    }
    torch.save(metadata, os.path.join(save_dir, 'metadata.pt'))

    print(f"\nDone! Fisher G matrices saved to {save_dir}")
    return save_dir


def get_dataloader(args, tokenizer, seqlen=2048):
    from data_utils import get_loaders
    result = get_loaders(
        args.calib_data,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=seqlen,
    )
    return result[0] if isinstance(result, tuple) else result


def main():
    parser = argparse.ArgumentParser(description='Pre-compute gradients for KronQ')

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--calib_data', type=str, default='wikitext2')
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--fp32_had', action='store_true')
    parser.add_argument('--rotate_mode', type=str, default='hadamard')

    parser.add_argument('--fisher_samples', type=int, default=1,
                        help='Number of Fisher samples per batch')

    args = parser.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dev}")
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"({torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.1f} GB)")

    print(f"Loading model: {args.model}")

    # device_map='auto' distributes across available GPUs.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.seqlen = args.seqlen

    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)

    # Always: strip accelerate hooks + consolidate to GPU 0 + enable grad checkpoint.
    # (For models too large for one GPU, would need save+reload.)
    if hasattr(model, 'hf_device_map'):
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)
        del model.hf_device_map
    model = model.to('cuda:0')

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if args.rotate:
        # Add activation-quant wrappers so calibration matches the quantized forward
        quant_utils.add_actquant(model)
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if 'down_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size // model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had

        print("Rotation applied!")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataloader = get_dataloader(args, tokenizer, args.seqlen)

    print(f"Using Fisher sampling with {args.fisher_samples} samples per batch")
    precompute_fisher_gradients(model, dataloader, dev, args, save_dir=args.output)


if __name__ == '__main__':
    main()