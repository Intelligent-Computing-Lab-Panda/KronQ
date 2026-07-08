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
"""
Decode-latency micro-benchmark for KronQ / GPTAQ / FP16 baselines.

Loads a model in the same way main.py does (rotate -> fuse_layer_norms ->
rotate_model -> add_actquant -> online Hadamard setup -> load_state_dict),
then runs model.generate() under controlled conditions and reports
TPOT (time-per-output-token), throughput (tok/s), and peak GPU memory.

Fake-quant path: weights are stored as fp16 with quantization error baked
in. Measures the rotation/wrapper overhead but NOT real int4 memory savings.
For real int4 memory savings, use the packed deployment path
(main.py --save_packed to pack + runtime/packed_io.py to load).

Usage:
  # FP16 baseline
  python bench_decode.py --model meta-llama/Llama-2-7b-hf --method fp16

  # GPTAQ baseline (loads pre-quantized checkpoint)
  python bench_decode.py --model meta-llama/Llama-2-7b-hf \\
      --load_qmodel_path /path/to/gptaq_w2.pth \\
      --method gptaq

  # KronQ (loads pre-quantized checkpoint; --rotate not needed because
  # BiIP rotation is baked into the weights at quantization time)
  python bench_decode.py --model meta-llama/Llama-2-7b-hf \\
      --load_qmodel_path /path/to/kronq_w2.pth \\
      --method kronq
"""

import argparse
import json
import os
import statistics
import time

import torch
import transformers

import graph_wrapper
import hadamard_utils
import biip_linear
import model_utils
import quant_utils
import rotation_utils
import utils


def build_model(args):
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=False)

    if args.load_qmodel_path:
        # Match main.py's path: add_actquant before load_state_dict so the
        # ActQuantWrapper-wrapped names exist in the destination state dict.
        quant_utils.add_actquant(model)
        if args.rotate:
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
        save_dict = torch.load(args.load_qmodel_path, map_location='cpu')
        # KronQ ckpts saved with --save_unrotated_kronq carry per-Linear biip_*
        # buffers in the state_dict; pre-register them on the model so they load.
        # No-op for GPTQ/GPTAQ ckpts.
        biip_linear.register_biip_buffers_from_state_dict(model, save_dict["model"])
        model.load_state_dict(save_dict["model"], strict=False)
        del save_dict
        # If we just loaded a KronQ-unrotated ckpt, wrap each Linear with online
        # BiIP. No-op when no biip_* buffers were registered.
        n_wrapped = biip_linear.wrap_linears_with_biip(model)
        if n_wrapped > 0:
            print(f"Wrapped {n_wrapped} Linears with online BiIP")

    return model


def apply_graph_wrapper(model, device=0):
    """Convert a loaded model to a CUDA-graph-wrapped version (in-place via
    __class__ swap). Avoids re-running __init__ on the already-loaded weights."""
    GraphCls = graph_wrapper.get_graph_wrapper(type(model), device=device)
    model.__class__ = GraphCls
    model.built_graph = False
    model.graph_device = device
    return model


def time_single_token_forward(model, dev, vocab_size, batch_size, num_warmup, num_trials):
    """Single-token forward bench (QuIP#-style single-token methodology).
    Times one forward of shape (batch, 1) with use_cache=False per trial.
    Returns list of per-trial seconds."""
    token = torch.randint(0, vocab_size, (batch_size, 1), device=dev)
    for _ in range(num_warmup):
        with torch.no_grad():
            model(token, use_cache=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(token, use_cache=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def time_generate(model, tokenizer, prompt_ids, num_gen_tokens, num_warmup, num_trials):
    """Returns list of per-trial total wall-clock seconds for generation."""
    # Greedy decoding to remove sampling variance.
    gen_kwargs = dict(
        max_new_tokens=num_gen_tokens,
        min_new_tokens=num_gen_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 0,
    )

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(prompt_ids, **gen_kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(prompt_ids, **gen_kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        # Sanity: generated exactly num_gen_tokens new tokens
        assert out.shape[1] == prompt_ids.shape[1] + num_gen_tokens, \
            f"Expected {prompt_ids.shape[1] + num_gen_tokens} tokens, got {out.shape[1]}"
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--load_qmodel_path', type=str, default=None,
                        help='Path to .pth saved by main.py --save_qmodel_path. Omit for FP16.')
    parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=False,
                        help='Must match the rotate setting used at quantization time.')
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--method', type=str, default='unspecified',
                        help='Label for the row in output (fp16 / gptaq / kronq).')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--prompt_tokens', type=int, default=64,
                        help='Length of input prompt in tokens.')
    parser.add_argument('--num_gen_tokens', type=int, default=256)
    parser.add_argument('--num_warmup', type=int, default=3)
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Optional path; appends one row per run.')
    parser.add_argument('--bench_mode', type=str, default='generate',
                        choices=['generate', 'single_tok'],
                        help='generate: full model.generate(max_new_tokens=N), '
                             'TPOT averaged. single_tok: one forward(shape (B,1), '
                             'use_cache=False) per trial; reports per-call '
                             'latency at batch size 1.')
    parser.add_argument('--use_cuda_graph', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Wrap model in CUDA graph (single_tok mode only).')
    parser.add_argument('--simulate_lowbit_storage', type=int, default=0,
                        choices=[0, 2, 4],
                        help='If 2 or 4, compress every BiIP-wrapped Linear to '
                             'packed int<bits> storage after load. Demonstrates '
                             'real-quant memory savings. Forward dequantizes on '
                             'the fly (no fused kernel — slower, but peak '
                             'memory after compression reflects actual deploy).')
    parser.add_argument('--simulate_input_only', action='store_true',
                        help='Strip output-side Hadamard + output rescale from '
                             'every BiIPLinear after compress. Produces garbage '
                             'logits but TPOT is representative of an input-only '
                             '(original-QuIP-style) deploy with our kernel.')
    parser.add_argument('--simulate_no_rotation', action='store_true',
                        help='Strip BOTH input- and output-side Hadamard + '
                             'rescale from every BiIPLinear, leaving only the '
                             'per-channel dequant scale/zero + fused matvec. '
                             'Produces garbage logits but TPOT is representative '
                             'of a rotation-free PTQ (GPTQ/GPTAQ-style) deploy '
                             'under the identical kernel/graph/low-bit tier.')
    args = parser.parse_args()

    dev = utils.DEV
    print(f"Device: {dev}")
    print(f"Method: {args.method}")
    print(f"Model:  {args.model}")
    print(f"Load:   {args.load_qmodel_path or '(none, FP16 baseline)'}")
    print(f"Rotate: {args.rotate}")
    print(f"Batch:  {args.batch_size}   Prompt: {args.prompt_tokens} tok   Gen: {args.num_gen_tokens} tok")

    # Build & move model
    model = build_model(args)
    model.to(dev)
    if args.simulate_lowbit_storage > 0:
        from lowbit_utils import compress_model_to_lowbit
        # Re-open ckpt to grab w_quantizers (the original scale/zero) so
        # compression uses the correct values, not inferred from quant'd W.
        w_q = None
        if args.load_qmodel_path:
            _sd = torch.load(args.load_qmodel_path, map_location='cpu')
            w_q = _sd.get("w_quantizers", None)
            del _sd
        n_comp, max_err = compress_model_to_lowbit(model, args.simulate_lowbit_storage, w_quantizers=w_q)
        print(f"Compressed {n_comp} Linears to {args.simulate_lowbit_storage}-bit packed (max recon err {max_err:.2e})")
        torch.cuda.empty_cache()
    if args.simulate_input_only:
        # Strip output-side rotation/rescale from every BiIPLinear. Forward will
        # skip stages 4+5 (output Hadamard + output rescale). Logits are wrong;
        # purpose is wall-clock measurement of an input-only-rotation deploy.
        from biip_linear import BiIPLinear
        n_stripped = 0
        for module in model.modules():
            if isinstance(module, BiIPLinear):
                for attr in ('biip_SV', 'biip_output_mul', 'biip_scaleG',
                             '_had_right', '_K_right'):
                    if hasattr(module, attr):
                        delattr(module, attr)
                n_stripped += 1
        torch.cuda.empty_cache()
        print(f"Stripped output-side rotation from {n_stripped} BiIPLinears "
              f"(input-only simulation)")
    if args.simulate_no_rotation:
        # Strip BOTH input- and output-side rotation/rescale from every
        # BiIPLinear. Forward keeps only the packed-low-bit dequant + matvec
        # with per-channel scale/zero (stages 1, 4, 5 all skip). Logits are
        # wrong; purpose is wall-clock measurement of a rotation-free PTQ
        # (GPTQ/GPTAQ-style) deploy under the identical kernel/graph tier.
        from biip_linear import BiIPLinear
        n_stripped = 0
        for module in model.modules():
            if isinstance(module, BiIPLinear):
                for attr in ('biip_SU', 'biip_input_mul', 'biip_scaleH',
                             '_had_left_T', '_K_left',
                             'biip_SV', 'biip_output_mul', 'biip_scaleG',
                             '_had_right', '_K_right'):
                    if hasattr(module, attr):
                        delattr(module, attr)
                n_stripped += 1
        torch.cuda.empty_cache()
        print(f"Stripped BOTH-side rotation from {n_stripped} BiIPLinears "
              f"(rotation-free / GPTQ-style simulation)")
    if args.use_cuda_graph:
        if args.bench_mode != 'single_tok':
            raise ValueError("--use_cuda_graph only supported in single_tok bench_mode (variable shapes in generate() can't be captured).")
        model = apply_graph_wrapper(model, device=dev.index if hasattr(dev, 'index') else 0)
        print("Applied CUDA graph wrapper")
    torch.cuda.synchronize()
    print(f"Model loaded. Param count: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B")

    torch.manual_seed(args.seed)
    vocab_size = model.config.vocab_size

    # Reset memory stats just before generation so we capture decode-only peak.
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize()

    if args.bench_mode == 'single_tok':
        # Each "trial" is exactly 1 forward = 1 decode step; reports per-call
        # latency at batch size 1 (aggregation below uses TPOT = trial time).
        times = time_single_token_forward(
            model, dev, vocab_size=vocab_size, batch_size=args.batch_size,
            num_warmup=args.num_warmup, num_trials=args.num_trials,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, token=args.hf_token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # Fixed pseudo-random prompt for reproducibility. Using vocab_size-aware
        # random IDs rather than meaningful text — what we measure is the
        # per-step forward cost, which is independent of token content.
        prompt_ids = torch.randint(0, vocab_size, (args.batch_size, args.prompt_tokens), device=dev)
        times = time_generate(
            model, tokenizer, prompt_ids,
            num_gen_tokens=args.num_gen_tokens,
            num_warmup=args.num_warmup,
            num_trials=args.num_trials,
        )
        tokens_per_trial = args.batch_size * args.num_gen_tokens

    peak_mem_bytes = torch.cuda.max_memory_allocated(dev)
    peak_mem_gib = peak_mem_bytes / (1024 ** 3)

    # Aggregate
    if args.bench_mode == 'single_tok':
        # Trial time = TPOT for one decoded token (1-step forward).
        tpot_per_trial_ms = [t * 1000.0 for t in times]
        throughput_per_trial = [(1.0 / t) for t in times]
    else:
        tpot_per_trial_ms = [(t / args.num_gen_tokens) * 1000.0 for t in times]
        throughput_per_trial = [(tokens_per_trial / t) for t in times]

    tpot_median = statistics.median(tpot_per_trial_ms)
    tpot_p95 = sorted(tpot_per_trial_ms)[int(0.95 * len(tpot_per_trial_ms))] if len(tpot_per_trial_ms) > 1 else tpot_per_trial_ms[0]
    tput_median = statistics.median(throughput_per_trial)

    print()
    print(f"Trials: {len(times)}")
    print(f"TPOT (median): {tpot_median:.3f} ms/token")
    print(f"TPOT (p95):    {tpot_p95:.3f} ms/token")
    print(f"Throughput:    {tput_median:.2f} tok/s")
    print(f"Peak VRAM:     {peak_mem_gib:.3f} GiB")
    print()

    result = dict(
        method=args.method,
        model=args.model,
        batch_size=args.batch_size,
        prompt_tokens=args.prompt_tokens,
        num_gen_tokens=args.num_gen_tokens,
        num_trials=args.num_trials,
        rotate=args.rotate,
        tpot_median_ms=tpot_median,
        tpot_p95_ms=tpot_p95,
        throughput_tok_per_s=tput_median,
        peak_vram_gib=peak_mem_gib,
        per_trial_seconds=times,
    )
    print(json.dumps(result, indent=2))

    if args.output_csv:
        write_header = not os.path.exists(args.output_csv)
        with open(args.output_csv, 'a') as f:
            if write_header:
                f.write("method,model,batch_size,prompt_tokens,num_gen_tokens,num_trials,"
                        "rotate,bench_mode,cuda_graph,lowbit,tpot_median_ms,tpot_p95_ms,"
                        "throughput_tok_per_s,peak_vram_gib\n")
            f.write(f"{args.method},{args.model},{args.batch_size},"
                    f"{args.prompt_tokens},{args.num_gen_tokens},{args.num_trials},"
                    f"{args.rotate},{args.bench_mode},{args.use_cuda_graph},"
                    f"{args.simulate_lowbit_storage},"
                    f"{tpot_median:.4f},{tpot_p95:.4f},"
                    f"{tput_median:.4f},{peak_mem_gib:.4f}\n")
        print(f"Appended to {args.output_csv}")


if __name__ == '__main__':
    main()
