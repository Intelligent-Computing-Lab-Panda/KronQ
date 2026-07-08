"""Fused dequant + matvec Triton kernel for packed int2/int4 weights.

For batch=1 single-token decode:
  y[m] = sum_n (codes[m, n] - zero[m]) * scale[m] * x[n]
  where codes are stored packed (uint8) along input dim.

Avoids materializing the full fp16 W tensor on every forward (the bottleneck
in the naive `dequant_packed` → `F.linear` path).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _matvec_int4_kernel(
    x_ptr, packed_ptr, scale_ptr, zero_ptr, y_ptr,
    M, N, stride_pm,
    BLOCK_M: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,  # packed bytes per iteration (= BLOCK_K // 2)
):
    """Stay in fp16 throughout inner loop (better tensor-core use + less reg
    pressure). Apply scale/zero only at the end via algebra:
        y = sum (codes - zero) * scale * x = scale * (sum codes*x - zero * sum x)
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0)
    zero = tl.load(zero_ptr + rows, mask=row_mask, other=0.0)

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)   # accumulator stays fp32 for precision
    sum_x = tl.zeros([1], dtype=tl.float32)
    N_PACKED = N // 2

    for kp_start in range(0, N_PACKED, BLOCK_PACKED):
        pcol = kp_start + tl.arange(0, BLOCK_PACKED)
        pcol_mask = pcol < N_PACKED
        p_offsets = rows[:, None] * stride_pm + pcol[None, :]
        full_mask = row_mask[:, None] & pcol_mask[None, :]
        packed = tl.load(packed_ptr + p_offsets, mask=full_mask, other=0)

        low_codes  = (packed & 0xF).to(tl.float16)
        high_codes = ((packed >> 4) & 0xF).to(tl.float16)

        x_off = 2 * kp_start + 2 * tl.arange(0, BLOCK_PACKED)
        x_low  = tl.load(x_ptr + x_off,     mask=x_off < N,       other=0.0)
        x_high = tl.load(x_ptr + x_off + 1, mask=(x_off + 1) < N, other=0.0)

        acc += tl.sum((low_codes * x_low[None, :]).to(tl.float32), axis=1)
        acc += tl.sum((high_codes * x_high[None, :]).to(tl.float32), axis=1)
        sum_x += tl.sum(x_low.to(tl.float32) + x_high.to(tl.float32))

    # y = scale * (acc - zero * sum_x)
    out = scale.to(tl.float32) * (acc - zero.to(tl.float32) * sum_x)
    tl.store(y_ptr + rows, out.to(tl.float16), mask=row_mask)


@triton.jit
def _matvec_int2_kernel(
    x_ptr, packed_ptr, scale_ptr, zero_ptr, y_ptr,
    M, N, stride_pm,
    BLOCK_M: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,  # packed bytes per iteration (= BLOCK_K // 4)
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    zero = tl.load(zero_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    N_PACKED = N // 4

    for kp_start in range(0, N_PACKED, BLOCK_PACKED):
        pcol = kp_start + tl.arange(0, BLOCK_PACKED)
        pcol_mask = pcol < N_PACKED
        p_offsets = rows[:, None] * stride_pm + pcol[None, :]
        full_mask = row_mask[:, None] & pcol_mask[None, :]
        packed = tl.load(packed_ptr + p_offsets, mask=full_mask, other=0)

        c0 = (packed & 0x3).to(tl.float32) - zero[:, None]
        c1 = ((packed >> 2) & 0x3).to(tl.float32) - zero[:, None]
        c2 = ((packed >> 4) & 0x3).to(tl.float32) - zero[:, None]
        c3 = ((packed >> 6) & 0x3).to(tl.float32) - zero[:, None]

        x_off = 4 * kp_start + 4 * tl.arange(0, BLOCK_PACKED)
        x0 = tl.load(x_ptr + x_off,     mask=x_off < N,       other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x_off + 1, mask=(x_off+1) < N,   other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_off + 2, mask=(x_off+2) < N,   other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + x_off + 3, mask=(x_off+3) < N,   other=0.0).to(tl.float32)

        acc += tl.sum(c0 * x0[None, :], axis=1)
        acc += tl.sum(c1 * x1[None, :], axis=1)
        acc += tl.sum(c2 * x2[None, :], axis=1)
        acc += tl.sum(c3 * x3[None, :], axis=1)

    acc = acc * scale
    tl.store(y_ptr + rows, acc.to(tl.float16), mask=row_mask)


def fused_dequant_matvec(x, packed, scale, zero, bits, out_dim_last):
    """x: (..., N) fp16. packed: (M, N//packing) uint8. scale, zero: (M,) fp16.
    Returns y: (..., M) fp16.
    Currently only batch_size effectively == 1 (last dim is reduced; leading dims
    are treated as independent matvecs — but kernel is matvec-only, so we squeeze
    to (N,))."""
    # Flatten leading dims; the matvec kernel handles batch == 1 only
    orig_shape = x.shape
    if x.ndim > 1:
        bsz = 1
        for d in x.shape[:-1]:
            bsz *= d
        if bsz != 1:
            # Fallback to non-fused for batch > 1
            from lowbit_utils import dequant_packed
            W = dequant_packed(packed, scale, zero, bits, out_dim_last)
            return (x @ W.T).to(x.dtype)
        x = x.reshape(-1)
    M = scale.shape[0]
    N = x.shape[-1]
    y = torch.empty(M, dtype=torch.float16, device=x.device)
    BLOCK_M = 8
    grid = (triton.cdiv(M, BLOCK_M),)
    stride_pm = packed.stride(0)
    if bits == 4:
        BLOCK_PACKED = 512
        _matvec_int4_kernel[grid](
            x, packed, scale, zero, y, M, N, stride_pm,
            BLOCK_M=BLOCK_M, BLOCK_PACKED=BLOCK_PACKED,
            num_warps=4, num_stages=2,
        )
    elif bits == 2:
        BLOCK_PACKED = 512
        _matvec_int2_kernel[grid](
            x, packed, scale, zero, y, M, N, stride_pm,
            BLOCK_M=BLOCK_M, BLOCK_PACKED=BLOCK_PACKED,
            num_warps=4, num_stages=2,
        )
    else:
        raise ValueError(f"bits must be 2 or 4, got {bits}")
    return y.reshape(*orig_shape[:-1], M)
