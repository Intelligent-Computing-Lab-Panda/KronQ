"""Packed int2/int4 weight storage + dequant utilities.

Storage scheme: per-output-channel asymmetric quantization.
  W_fp[i, j] = (codes[i, j] - zero[i]) * scale[i]
where codes ∈ uint8 in [0, 2^bits - 1].

Packed buffers stored as uint8 along the last (input) dim:
  bits=4: 2 codes per byte → packed shape (m, n // 2)
  bits=2: 4 codes per byte → packed shape (m, n // 4)

Why this design:
  - per-out-channel matches what `WeightQuantizer(perchannel=True)` produces
  - packing along input dim keeps unpacking cache-friendly per output row
  - storing only ±codes (no fp16 W) gives the actual W2/W4 storage reduction:
        fp16 13 GiB → W4 ~3.4 GiB → W2 ~1.7 GiB for L2-7B
"""
import torch
import logging


def pack_codes(codes, bits):
    """codes: (m, n) integer tensor (any int dtype) with values in [0, 2^bits-1].
    Returns uint8 tensor of shape (m, n // codes_per_byte)."""
    if bits not in (2, 4):
        raise ValueError(f"Only bits ∈ {{2, 4}} supported, got {bits}")
    codes_per_byte = 8 // bits
    if codes.shape[-1] % codes_per_byte != 0:
        raise ValueError(f"Last dim {codes.shape[-1]} not divisible by {codes_per_byte}")
    codes = codes.to(torch.uint8).contiguous()
    if bits == 4:
        # 2 codes per byte: low 4 bits = codes[..., 0::2], high 4 bits = codes[..., 1::2]
        packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    else:  # bits == 2
        # 4 codes per byte
        packed = (
            codes[..., 0::4]
            | (codes[..., 1::4] << 2)
            | (codes[..., 2::4] << 4)
            | (codes[..., 3::4] << 6)
        )
    return packed


def unpack_codes(packed, bits, out_dim_last):
    """packed: (m, n_packed) uint8 tensor. Returns (m, out_dim_last) uint8 codes."""
    if bits not in (2, 4):
        raise ValueError(f"Only bits ∈ {{2, 4}} supported, got {bits}")
    if bits == 4:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        out = torch.stack([low, high], dim=-1).flatten(start_dim=-2)
    else:  # bits == 2
        a = packed & 0x03
        b = (packed >> 2) & 0x03
        c = (packed >> 4) & 0x03
        d = (packed >> 6) & 0x03
        out = torch.stack([a, b, c, d], dim=-1).flatten(start_dim=-2)
    if out.shape[-1] != out_dim_last:
        raise RuntimeError(f"Unpacked dim {out.shape[-1]} != expected {out_dim_last}")
    return out


def dequant_packed(packed, scale, zero, bits, out_dim_last):
    """packed: (m, n_packed) uint8. scale, zero: (m,) per-output-channel.
    Returns fp16 (or scale.dtype) dequantized W of shape (m, out_dim_last).
    Math: W = (unpack(packed) - zero[:, None]) * scale[:, None]"""
    codes = unpack_codes(packed, bits, out_dim_last)
    out_dtype = scale.dtype
    codes_f = codes.to(out_dtype)
    return (codes_f - zero[:, None]) * scale[:, None]


def extract_codes_from_dequant_W(W_fp16, scale, zero, bits):
    """Reverse the dequant: given W_fp16 = (codes - zero) * scale, recover codes.
    Asymmetric per-output-channel.

    W_fp16: (m, n) — already dequantized (output of WeightQuantizer.quantize())
    scale, zero: (m,) per-out-channel
    Returns uint8 codes (m, n) clamped to [0, 2^bits - 1].
    """
    max_code = (1 << bits) - 1
    # codes ≈ round(W / scale + zero)
    codes = torch.round(W_fp16 / scale[:, None] + zero[:, None])
    codes = codes.clamp(0, max_code).to(torch.uint8)
    return codes


def expand_group_to_cols(group_t, gid, n):
    """group_t: (m, n_groups), gid: (n,) int -> per-column (m, n) by gather."""
    gid = gid.to(device=group_t.device, dtype=torch.long)
    return group_t.index_select(1, gid)  # (m, n)


def extract_codes_from_dequant_W_group(W_fp16, group_scale, group_zero, gid, bits):
    """Per-group code recovery. group_scale/group_zero: (m, n_groups);
    gid: (n,) column->group. Returns uint8 codes (m, n)."""
    m, n = W_fp16.shape
    max_code = (1 << bits) - 1
    scale_c = expand_group_to_cols(group_scale, gid, n)  # (m, n)
    zero_c = expand_group_to_cols(group_zero, gid, n)
    codes = torch.round(W_fp16 / scale_c + zero_c)
    codes = codes.clamp(0, max_code).to(torch.uint8)
    return codes


def dequant_packed_group(packed, group_scale, group_zero, gid, bits, out_dim_last):
    """Per-group dequant. group_scale/group_zero: (m, n_groups); gid: (n,)."""
    codes = unpack_codes(packed, bits, out_dim_last)
    out_dtype = group_scale.dtype
    scale_c = expand_group_to_cols(group_scale, gid, out_dim_last)
    zero_c = expand_group_to_cols(group_zero, gid, out_dim_last)
    return (codes.to(out_dtype) - zero_c) * scale_c


def compress_model_to_lowbit(model, bits, w_quantizers=None):
    """Walk model, find all Linears with biip_* buffers, compress them to
    packed low-bit. Returns (num_compressed, max_reconstruction_err).

    `w_quantizers`: optional dict of module path ('model.layers.0.self_attn.q_proj')
    → WeightQuantizer carrying the original quantization scale/zero. If None,
    scale/zero are inferred from the dequant'd W — only exact when codes span
    the full [0, maxq] range, which they often don't post-asym-quant."""
    import torch.nn as nn
    n_compressed = 0
    max_err = 0.0
    # Build a map from Linear instance → its full path in the model.
    # This lets us look up w_quantizers entries that are keyed by path.
    linear_path = {}
    def _walk(mod, prefix):
        for name, child in mod.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                linear_path[id(child)] = full
            else:
                _walk(child, full)
    _walk(model, "")

    for module in model.modules():
        if isinstance(module, nn.Linear) and getattr(module, 'biip_SU', None) is not None:
            scale_in = None; zero_in = None
            group_scale = None; group_zero = None; group_gid = None
            if w_quantizers is not None:
                full = linear_path.get(id(module), None)
                # w_quantizers keys come in 2 flavors per how fasterquant saves:
                # 'model.layers.X.attr.module' or 'model.layers.X.attr'. Try both.
                candidates = []
                if full is not None:
                    candidates.append(full)
                    if full.endswith('.module'):
                        candidates.append(full[:-len('.module')])
                for k in candidates:
                    if k in w_quantizers:
                        q = w_quantizers[k]
                        # Per-group (g128): fasterquant stored group_scale/zero/gid.
                        gs = getattr(q, 'group_scale', None)
                        if gs is not None:
                            group_scale = gs.to(module.weight.device)
                            group_zero = q.group_zero.to(module.weight.device)
                            group_gid = q.group_gid.to(module.weight.device)
                        else:
                            scale_in = q.scale.flatten().to(module.weight.device)
                            zero_in = q.zero.flatten().to(module.weight.device)
                        break
            err = compress_linear_to_lowbit(module, bits, scale_in, zero_in,
                                            group_scale=group_scale,
                                            group_zero=group_zero,
                                            group_gid=group_gid)
            max_err = max(max_err, err)
            n_compressed += 1
    return n_compressed, max_err


def compress_linear_to_lowbit(linear, bits, scale=None, zero=None,
                              group_scale=None, group_zero=None, group_gid=None):
    """Take a Linear with fp16 weight (assumed already dequant'd from low-bit
    quantization with per-output-channel scale/zero), extract integer codes,
    pack them, and register as buffers on the Linear:
        biip_w_codes  (uint8, shape (m, n // codes_per_byte))
        biip_w_scale  (fp16, shape (m,)  per-row  OR  (m, n_groups) per-group)
        biip_w_zero   (fp16, same shape as biip_w_scale)
        biip_w_gid    (int32, (n,))      only when per-group
    The Linear's `weight.data` is shrunk to 1×1.

    `scale`/`zero` (1D, length m): original quantization params, needed for exact
    code recovery (a quantized W's data range underestimates the [0, maxq] range).
    `group_scale`/`group_zero` (m, n_groups) + `group_gid` (n,): per-group (g128)
    path — column j uses scale[:, gid[j]], bit-exact under act_order permutation.
    """
    W = linear.weight.data.float()  # cast for analysis
    m, n = W.shape
    max_code = (1 << bits) - 1

    # ---- Per-group (g128) path ----
    if group_scale is not None and group_zero is not None and group_gid is not None:
        group_scale = group_scale.float().to(W.device)
        group_zero = group_zero.float().to(W.device)
        group_gid = group_gid.to(device=W.device, dtype=torch.int32)
        if hasattr(linear, '_ensure_had_kernels'):
            linear._ensure_had_kernels()
        codes = extract_codes_from_dequant_W_group(W, group_scale, group_zero,
                                                   group_gid, bits)
        packed = pack_codes(codes, bits)
        target_dtype = linear.weight.data.dtype
        target_dev = linear.weight.data.device
        W_recon = dequant_packed_group(
            packed, group_scale.to(torch.float16), group_zero.to(torch.float16),
            group_gid, bits, n)
        recon_err = (W.to(torch.float16) - W_recon).abs().max().item()
        linear.register_buffer('biip_w_codes', packed.to(target_dev))
        linear.register_buffer('biip_w_scale',
                               group_scale.to(device=target_dev, dtype=target_dtype).contiguous())
        linear.register_buffer('biip_w_zero',
                               group_zero.to(device=target_dev, dtype=target_dtype).contiguous())
        linear.register_buffer('biip_w_gid', group_gid.to(target_dev).contiguous())
        linear.biip_w_bits = bits
        linear.biip_w_out_dim = n
        _register_biip_muls(linear, packed, bits, target_dev, target_dtype)
        linear.weight.data = torch.empty(1, 1, dtype=target_dtype, device=target_dev)
        return recon_err

    if scale is None or zero is None:
        # Fallback: infer from data range. Only correct if codes span [0, max_code].
        xmin = torch.minimum(W.min(1).values, torch.zeros(m, device=W.device))
        xmax = torch.maximum(W.max(1).values, torch.zeros(m, device=W.device))
        tmp = (xmin == 0) & (xmax == 0)
        xmin = xmin.masked_fill(tmp, -1.0)
        xmax = xmax.masked_fill(tmp,  1.0)
        scale = (xmax - xmin).clamp(min=1e-5) / max_code
        zero = torch.round(-xmin / scale)
    else:
        scale = scale.float().to(W.device)
        zero = zero.float().to(W.device)

    # Must run before the weight is shrunk to 1×1 below:
    # _ensure_had_kernels reads `weight.shape` to size the Hadamard kernels.
    if hasattr(linear, '_ensure_had_kernels'):
        linear._ensure_had_kernels()

    # Extract codes via inverse dequant
    codes = extract_codes_from_dequant_W(W, scale, zero, bits)
    packed = pack_codes(codes, bits)

    # Verify reconstruction
    W_recon = dequant_packed(packed, scale.to(torch.float16),
                              zero.to(torch.float16), bits, n)
    recon_err = (W.to(torch.float16) - W_recon).abs().max().item()
    if recon_err > 1e-2:
        logging.warning(f"Packed reconstruction error {recon_err:.2e} exceeds 1e-2")

    # Register buffers
    target_dtype = linear.weight.data.dtype
    target_dev = linear.weight.data.device
    linear.register_buffer('biip_w_codes', packed.to(target_dev))
    linear.register_buffer('biip_w_scale', scale.to(device=target_dev, dtype=target_dtype))
    linear.register_buffer('biip_w_zero', zero.to(device=target_dev, dtype=target_dtype))
    linear.biip_w_bits = bits
    linear.biip_w_out_dim = n

    _register_biip_muls(linear, packed, bits, target_dev, target_dtype)

    # Replace weight storage with a 1×1 dummy via .data assignment to free fp16.
    linear.weight.data = torch.empty(1, 1, dtype=target_dtype, device=target_dev)

    return recon_err


def _register_biip_muls(linear, packed, bits, target_dev, target_dtype):
    # Precompute combined rescale-times-sign so BiIPLinear.forward does one
    # elementwise multiply per side instead of two:
    #   biip_input_mul[n]  = biip_SU[n]  / biip_scaleH[n]
    #   biip_output_mul[m] = biip_SV[m]  / biip_scaleG[m]
    in_features  = packed.shape[1] * (8 // bits)   # = n
    out_features = packed.shape[0]                  # = m
    biip_SU = getattr(linear, 'biip_SU', None)
    biip_scaleH = getattr(linear, 'biip_scaleH', None)
    biip_SV = getattr(linear, 'biip_SV', None)
    biip_scaleG = getattr(linear, 'biip_scaleG', None)
    one_in  = torch.ones(in_features,  device=target_dev, dtype=target_dtype)
    one_out = torch.ones(out_features, device=target_dev, dtype=target_dtype)
    input_mul  = (biip_SU.to(target_dtype)  if biip_SU  is not None else one_in)
    if biip_scaleH is not None:
        input_mul = input_mul / biip_scaleH.to(target_dtype)
    output_mul = (biip_SV.to(target_dtype)  if biip_SV  is not None else one_out)
    if biip_scaleG is not None:
        output_mul = output_mul / biip_scaleG.to(target_dtype)
    linear.register_buffer('biip_input_mul',  input_mul.contiguous())
    linear.register_buffer('biip_output_mul', output_mul.contiguous())
