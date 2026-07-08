# --- repo path bootstrap ---
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not all(_os.path.isdir(_os.path.join(_d, s)) for s in ('common', 'runtime', 'calib')):
    _d = _os.path.dirname(_d)
for _s in ('common', 'runtime', 'calib'):
    _p = _os.path.join(_d, _s)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
"""Serialize / load a KronQ deploy model in PACKED int4/int2 form — the artifact
that goes to the HuggingFace Hub (~bits/16 the size of fp16; e.g. L2-7B W4 ≈ 3.6 GB
vs 13 GB fp16).

The deploy `BiIPLinear.forward` needs only these per-Linear tensors:
    biip_w_codes (uint8 packed), biip_w_scale, biip_w_zero,
    biip_input_mul, biip_output_mul   (SU/SV/scaleH/scaleG are folded into the muls)
plus scalars `bits` (global) and `out_dim` (= codes.shape[1] * 8//bits, derivable).
Everything else (embed_tokens, lm_head, norms, rotary) stays fp16.

  save_packed(model, out_dir, bits)   # call AFTER compress_model_to_lowbit(model, bits)
  load_packed(model, out_dir)         # model = fresh fp16 model w/ add_actquant applied
"""
import json
import torch
from safetensors.torch import save_file, load_file

# biip_SU / biip_SV MUST be kept: BiIPLinear.forward gates the Hadamard rotation
# on `biip_SU is not None` / `biip_SV is not None` (and reads .dim() to pick
# had-vs-kron mode). Their sign values are already folded into input_mul/output_mul,
# and the forward does NOT re-apply them when those muls are present. scaleH/scaleG
# are only read in the (un-taken) fallback branch, so they're safe to drop.
_DROP_BIIP = ('biip_scaleH', 'biip_scaleG')


def save_packed(model, out_dir, bits):
    """Write model.safetensors + kronq_packed_config.json. Returns (n_tensors, n_compressed)."""
    sd = model.state_dict()
    compressed = {k[: -len('.biip_w_codes')] for k in sd if k.endswith('.biip_w_codes')}
    keep = {}
    seen_ptrs = {}   # storage data_ptr -> first key kept (dedup shared tensors)
    for k, v in sd.items():
        # drop every 1x1 dummy weight left by compress (both the inner Linear's
        # `.module.weight` and any wrapper-level `.weight` view) — real weights
        # are never 1x1, so this is unambiguous.
        if k.endswith('.weight') and tuple(v.shape) == (1, 1):
            continue
        # drop SU/SV/scaleH/scaleG (already folded into input_mul/output_mul)
        if any(k.endswith('.' + s) for s in _DROP_BIIP):
            continue
        # Drop tensors that SHARE storage with an already-kept one (e.g. an
        # ActQuantWrapper exposes both `lm_head.weight` and `lm_head.module.weight`
        # backed by the same memory — safetensors rejects shared storage). The
        # dropped alias is reconstructed at load (strict=False) from its sibling /
        # the base model, identical for these un-quantized fp16 weights.
        try:
            ptr = v.untyped_storage().data_ptr() if v.numel() > 0 else None
        except Exception:
            ptr = v.storage().data_ptr() if v.numel() > 0 else None
        if ptr is not None and ptr in seen_ptrs:
            continue
        if ptr is not None:
            seen_ptrs[ptr] = k
        keep[k] = v.detach().contiguous().cpu()
    _os.makedirs(out_dir, exist_ok=True)
    save_file(keep, _os.path.join(out_dir, 'model.safetensors'))
    # Detect per-group (g128) packing: any biip_w_scale that is 2D (m, n_groups).
    grouped = any(k.endswith('.biip_w_scale') and v.dim() == 2 and v.shape[1] > 1
                  for k, v in sd.items())
    cfg = {'format': 'kronq-packed', 'bits': int(bits), 'n_compressed': len(compressed),
           'grouped': bool(grouped)}
    with open(_os.path.join(out_dir, 'kronq_packed_config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    return len(keep), len(compressed)


def load_packed(model, packed_dir):
    """Install packed weights into `model` (fresh fp16 model, add_actquant applied)
    and wrap Linears with online BiIP. Returns n_wrapped."""
    from biip_linear import register_biip_buffers_from_state_dict, wrap_linears_with_biip
    with open(_os.path.join(packed_dir, 'kronq_packed_config.json')) as f:
        cfg = json.load(f)
    bits = int(cfg['bits'])
    codes_per_byte = 8 // bits
    sd = load_file(_os.path.join(packed_dir, 'model.safetensors'))
    register_biip_buffers_from_state_dict(model, sd)
    model.load_state_dict(sd, strict=False)
    for module in model.modules():
        if getattr(module, 'biip_w_codes', None) is not None:
            module.biip_w_bits = bits
            module.biip_w_out_dim = module.biip_w_codes.shape[1] * codes_per_byte
    n = wrap_linears_with_biip(model)
    # Hadamard kernels read weight.shape -> init while the (full) weight is still
    # present, THEN shrink to a 1x1 dummy to free the fp16 weight memory.
    for module in model.modules():
        if getattr(module, 'biip_w_codes', None) is not None:
            if hasattr(module, '_ensure_had_kernels'):
                module._ensure_had_kernels()
            module.weight.data = torch.empty(1, 1, dtype=module.weight.dtype,
                                              device=module.weight.device)
    return n
