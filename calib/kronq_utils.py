"""KronQ weight quantization: Kronecker-factored (H_X ⊗ H_G) calibration with
bidirectional incoherence. The production --bi_calibration path."""

import math
import time
import copy
import torch
import torch.nn as nn
import os
import logging

import utils
import quant_utils
import model_utils

from incoherence import incoherence_preprocess_kronq, incoherence_postprocess_kronq
from inter_layer_mp import get_sublayer_bits

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_sublayer_gradient(grad_dir, layer_idx, sublayer_name):
    """Load pre-computed G matrix for a sublayer."""
    clean_name = sublayer_name.replace('.module', '')
    fname = clean_name.replace('.', '_') + '_G.pt'
    path = os.path.join(grad_dir, f"layer_{layer_idx}", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"G matrix file not found: {path}")
    G = torch.load(path)
    return G


def _incoh_group_allows(name):
    """Ablation switch: KRONQ_INCOH_GROUP=qk|no_qk|attn|mlp restricts BiIP to a
    sublayer group (plain GPTAQ elsewhere). Default 'all' applies BiIP everywhere."""
    grp = os.environ.get('KRONQ_INCOH_GROUP', 'all')
    if grp == 'all':
        return True
    is_qk = ('q_proj' in name) or ('k_proj' in name)
    return {'qk': is_qk, 'no_qk': not is_qk,
            'attn': 'self_attn' in name, 'mlp': 'mlp' in name}.get(grp, True)


class KronQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.G = None

        # GPTAQ: fp_inp buffer (set externally from FPInputsCache)
        self.fp_inp = []

    def set_G(self, G):
        """Set pre-computed gradient covariance."""
        self.G = G.float()

    def add_batch(self, inp, out, use_gptaq=True):
        """inp: degraded input x~ (from quantized forward pass);
        self.fp_inp[0]: full-precision input x_fp (from fp forward pass)."""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

        # only compute dXXT when using GPTAQ
        if use_gptaq:
            # GPTAQ: dXXT = E[(x_fp - x~) * x~^T]
            dX = self.fp_inp[0].float() * math.sqrt(2 / self.nsamples) - inp
            self.dXXT += dX.matmul(inp.t())
            del self.fp_inp[0]

    def fasterquant(
        self, blocksize=128, percdamp=0.01, groupsize=-1, actorder=False,
        static_groups=False, alpha=0.25, use_gptaq=True,
        incoh_rotate=False, incoh_mode='full', incoh_kernel='kron',
        save_unrotated=False,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        # ============ Prepare H ============
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.dXXT[:, dead] = 0

        # ============ Prepare G ============
        if incoh_rotate:
            G = self.G.clone().float().to(self.dev)
            del self.G
            dead_g = torch.diag(G) == 0
            G[dead_g, dead_g] = 1
        else:
            G = None
            if self.G is not None:
                del self.G

        # ============ Incoherence Preprocessing ============
        SU, SV, scaleH, scaleG = None, None, None, None
        dXXT = self.dXXT
        del self.dXXT

        if incoh_rotate:
            W, H, G, dXXT, SU, SV, scaleH, scaleG = incoherence_preprocess_kronq(
                W, H, G, dXXT, self.dev, mode=incoh_mode, kernel=incoh_kernel,
            )

        # Release H_G — algebraically cancels in the column-wise update
        del G

        # ============ Static groups ============
        if static_groups:
            groups = []
            for gi in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, gi:(gi + groupsize)])
                groups.append(quantizer)

        # ============ Actorder: column perm (H diag) + row perm (G diag) ============
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            dXXT = dXXT[perm][:, perm]
            invperm = torch.argsort(perm)

        # ============ Compute Hinv ============
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        # ============ GPTAQ P matrix ============
        if use_gptaq:
            P = alpha * ((dXXT @ Hinv.T).triu_(diagonal=1)) @ Hinv
        else:
            P = torch.zeros_like(Hinv)
        del dXXT

        if groupsize == -1:
            self.quantizer.find_params(W)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        # ============ Per-column scale/zero capture (for packed deploy) ============
        # Record per-column scale/zero (in permuted order) so the packed path
        # reproduces W bit-exactly; un-permuted and collapsed to (M, n_groups) below.
        capture_sz = (groupsize != -1)
        if capture_sz:
            col_scale = torch.zeros_like(W)
            col_zero = torch.zeros_like(W)
            # true group id per (permuted) column, filled in the loop
            col_group = torch.zeros(self.columns, dtype=torch.long, device=self.dev)

        # ============ Main GPTQ quantization loop ============
        torch.cuda.synchronize()
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2]

            for i in range(count):
                col_idx = i1 + i

                # --- Per-group quantizer params ---
                if groupsize != -1:
                    if not static_groups:
                        if col_idx % groupsize == 0:
                            self.quantizer.find_params(W[:, col_idx:(col_idx + groupsize)])
                    else:
                        idx = col_idx
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                w_orig = W1[:, i].clone()

                q_col = self.quantizer.quantize(W1[:, i].unsqueeze(1)).flatten()

                if capture_sz:
                    # quantizer.scale/zero are (M,1) for the active group/column
                    col_scale[:, col_idx] = self.quantizer.scale.flatten()
                    col_zero[:, col_idx] = self.quantizer.zero.flatten()
                    # record the column's TRUE group id (exact under act_order scatter)
                    if static_groups:
                        gidx = perm[col_idx] if actorder else col_idx
                        col_group[col_idx] = int(gidx) // groupsize
                    else:
                        col_group[col_idx] = col_idx // groupsize

                Q1[:, i] = q_col

                # Horizontal pass (GPTAQ)
                d_h = Hinv1[i, i]
                Losses1[:, i] = (w_orig - q_col) ** 2 / d_h ** 2
                err1 = (w_orig - q_col) / d_h

                W1[:, i:] -= (
                    err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    - w_orig.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                )

                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])

        torch.cuda.synchronize()

        # ============ Undo permutations ============
        if actorder:
            Q = Q[:, invperm]
            if capture_sz:
                col_scale = col_scale[:, invperm]
                col_zero = col_zero[:, invperm]
                col_group = col_group[invperm]

        # ============ Collapse per-column scale/zero -> per-group + gid ============
        # Compact (M, n_groups) scale/zero + column->group map, in original column
        # order. Stored on the quantizer for compress_model_to_lowbit (main.py).
        if capture_sz:
            n_groups = int(col_group.max().item()) + 1
            m = col_scale.shape[0]
            # all columns in a group share scale/zero, so scatter (last-write-wins) is exact
            group_scale = torch.zeros(m, n_groups, device=col_scale.device, dtype=col_scale.dtype)
            group_zero = torch.zeros(m, n_groups, device=col_zero.device, dtype=col_zero.dtype)
            idx = col_group.unsqueeze(0).expand(m, -1)  # (m, columns)
            group_scale.scatter_(1, idx, col_scale)
            group_zero.scatter_(1, idx, col_zero)
            self.quantizer.group_scale = group_scale
            self.quantizer.group_zero = group_zero
            self.quantizer.group_gid = col_group.to(torch.int32).contiguous()
            self.quantizer.groupsize = groupsize
            del col_scale, col_zero, col_group

        # ============ IncP Postprocess ============
        if incoh_rotate and not save_unrotated:
            # Default path: bake U, V, S_X, S_G back into Q (original coordinate space).
            Q = incoherence_postprocess_kronq(Q, SU, SV, scaleH, scaleG, self.dev, kernel=incoh_kernel)
        elif incoh_rotate and save_unrotated:
            # Deployment path: keep Q in rotated space and save the BiIP transforms
            # as buffers, applied online at inference (see BiIPLinear).
            target_dtype = self.layer.weight.data.dtype
            target_dev = self.layer.weight.data.device
            if SU is not None:
                self.layer.register_buffer('biip_SU', SU.to(device=target_dev, dtype=target_dtype))
            if SV is not None:
                self.layer.register_buffer('biip_SV', SV.to(device=target_dev, dtype=target_dtype))
            if scaleH is not None:
                self.layer.register_buffer('biip_scaleH', scaleH.to(device=target_dev, dtype=target_dtype))
            if scaleG is not None:
                self.layer.register_buffer('biip_scaleG', scaleG.to(device=target_dev, dtype=target_dtype))

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.G = None
        self.dXXT = None
        self.fp_inp = []
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


# ============================================================
# Pipeline: kronq_fwrd
# ============================================================

@torch.no_grad()
def kronq_fwrd(model, dataloader, dev, args):
    """Layer-wise KronQ quantization: dual forward pass for fp_inp/dXXT collection,
    per-sublayer G loading, and bidirectional incoherence.

    Relevant args: grad_dir (G matrices), incoh_rotate, incoh_mode (default 'full'),
    alpha (P-matrix scale, default 0.25), use_gptaq (P matrix on/off).
    """
    use_gptaq = getattr(args, 'use_gptaq', True)
    incoh_rotate = getattr(args, 'incoh_rotate', False)

    logging.info('-----KronQ Quantization-----')
    logging.info(f'  Horizontal comp: {"GPTAQ (P matrix)" if use_gptaq else "GPTQ (P=0)"}')
    logging.info(f'  Incoherence: {"ON" if incoh_rotate else "OFF"}')

    if incoh_rotate:
        if not hasattr(args, 'grad_dir') or args.grad_dir is None:
            raise ValueError("args.grad_dir required when incoh_rotate=True")
        if not os.path.exists(args.grad_dir):
            raise FileNotFoundError(f"Gradient directory not found: {args.grad_dir}")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, *args, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['layer_args'] = args
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            cache['layer_kwargs'] = kwargs  # for Gemma3 pe_global/pe_local etc.
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    quantizers = {}
    sequential = [
        ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
        ['self_attn.o_proj.module'],
        ['mlp.up_proj.module', 'mlp.gate_proj.module'],
        ['mlp.down_proj.module']
    ]

    if use_gptaq:
        fp_inputs_cache = model_utils.FPInputsCache(sequential)
    fp_inps = inps.clone()

    layer_args = cache.get('layer_args', ())

    def layer_forward(layer, inp):
        return layer(inp.unsqueeze(0), *layer_args, **cache['layer_kwargs'])[0]

    num_layers = len(layers)
    for i in range(num_layers):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        # ---- Collect fp_inps (act quant disabled), GPTAQ only ----
        if use_gptaq:
            bits_config = quant_utils.disable_act_quant(layer)
            fp_inputs_cache.add_hook(full)

            for j in range(args.nsamples):
                fp_inps[j] = layer_forward(layer, fp_inps[j])
            fp_inputs_cache.clear_hook()
            quant_utils.enable_act_quant(layer, bits_config)
        quant_start = time.time()
        # ---- Process each sequential group ----
        for names in sequential:
            subset = {n: full[n] for n in names}

            # ---- Step A: Create KronQ wrappers ----
            kaq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8

                kaq[name] = KronQ(subset[name])
                kaq[name].quantizer = quant_utils.WeightQuantizer()
                kaq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True,
                    sym=layer_weight_sym, mse=args.w_clip,
                )
                # only set fp_inp for GPTAQ
                if use_gptaq:
                    kaq[name].fp_inp = fp_inputs_cache.fp_cache[name]

            # ---- Step B: Load G matrices (if needed) ----
            if incoh_rotate:
                for name in kaq:
                    G = load_sublayer_gradient(args.grad_dir, i, name)
                    kaq[name].set_G(G)
                    del G
                    torch.cuda.empty_cache()

            # ---- Step C: Collect H and dXXT ----
            first_module_name = list(subset.keys())[0]

            def add_batch(name):
                def tmp(_, inp, out):
                    kaq[name].add_batch(inp[0].data, out.data, use_gptaq=use_gptaq)
                return tmp

            handle = subset[first_module_name].register_forward_hook(
                add_batch(first_module_name)
            )

            for j in range(args.nsamples):
                outs[j] = layer_forward(layer, inps[j])
            handle.remove()

            # Copy H and dXXT to siblings
            for name in subset:
                if name != first_module_name:
                    kaq[name].H = kaq[first_module_name].H
                    # only copy dXXT for GPTAQ
                    if use_gptaq:
                        kaq[name].dXXT = kaq[first_module_name].dXXT

            # ---- Step D: Quantize each sublayer ----
            for name in subset:
                layer_w_groupsize = args.w_groupsize

                if args.inter_layer_mp:
                    full_name = f'layers.{i}.{name}'
                    sublayer_bits = get_sublayer_bits(full_name, args.w_bits, args.inter_layer_mp)
                    sublayer_maxq = 2 ** sublayer_bits - 1
                    kaq[name].quantizer.maxq.fill_(sublayer_maxq)
                    logging.info(f'  Inter-layer MP: {name} -> {sublayer_bits}b (maxq={sublayer_maxq})')

                kaq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=args.act_order,
                    static_groups=args.static_groups,
                    alpha=getattr(args, 'alpha', 0.25),
                    incoh_rotate=getattr(args, 'incoh_rotate', False) and _incoh_group_allows(name),
                    incoh_mode=getattr(args, 'incoh_mode', 'full'),
                    incoh_kernel=getattr(args, 'incoh_kernel', 'kron'),
                    use_gptaq=use_gptaq,
                    save_unrotated=getattr(args, 'save_unrotated_kronq', False),
                )
                quantizers['model.layers.%d.%s' % (i, name)] = kaq[name].quantizer
                kaq[name].free()

            # With --save_unrotated_kronq, sublayers hold Q in rotated space; wrap
            # them now so the next group's forward uses the BiIPLinear online path
            # (else rotated Q is used as original-space weight -> NaN).
            if getattr(args, 'save_unrotated_kronq', False):
                from biip_linear import wrap_linears_with_biip
                wrap_linears_with_biip(layer)
        print("1 layer quantization took %.2f seconds" % (time.time() - quant_start))
        # ---- Update outputs for next layer ----
        for j in range(args.nsamples):
            outs[j] = layer_forward(layer, inps[j])
        if use_gptaq:
            fp_inputs_cache.clear_cache()
        layers[i] = layer.cpu()
        del layer
        del kaq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----KronQ Quantization Done-----\n')

    return quantizers
