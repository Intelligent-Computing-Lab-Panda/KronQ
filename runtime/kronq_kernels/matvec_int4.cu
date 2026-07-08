/*
 * Scalar int4 / int2 fused dequant + matvec for KronQ.
 *
 * Layout:
 *   x:        (N,)        fp16
 *   packed:   (M, N//2)   uint8 (int4: 2 codes per byte; low nibble = even index)
 *             OR (M, N//4) uint8 (int2: 4 codes per byte; low 2 bits = idx0)
 *   scale:    (M,)        fp16, per-output-row
 *   zero:     (M,)        fp16, per-output-row
 *   y:        (M,)        fp16
 *
 * Semantics (per output row i):
 *   y[i] = scale[i] * sum_j (codes[i, j] - zero[i]) * x[j]
 *
 * Kernel layout (v1, no tensor cores, no PTX tricks):
 *   - grid: (ceil(M / BLOCK_M),)
 *   - block: (BLOCK_THREADS,) = (256,) organized as 8 warps × 32 lanes
 *   - each block handles BLOCK_M = 8 output rows over full N
 *   - warps split N; each warp accumulates a partial sum per output row
 *   - reduction across lanes via warp shuffles, across warps via shared mem
 *
 * Dim assumptions (true for Llama/Qwen/etc; guarded in the host wrappers):
 *   v1 int4/int2: N % 16 == 0  (n_per_warp = N/8 must be even to cover all cols)
 *   v3 int4:      N % 32 == 0  (uint4 = 32 codes; 16-byte aligned row stride)
 *   v3 int2:      N % 64 == 0  (uint4 = 64 codes)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#define BLOCK_M 8
#define BLOCK_THREADS 256
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_THREADS / WARP_SIZE)  // 8

__global__ void matvec_int4_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scale,
    const __half* __restrict__ zero,
    const float* __restrict__ sum_x_ptr,  // pointer to single fp32 scalar (sum of x)
    __half* __restrict__ y,
    int M, int N
) {
    float sum_x = *sum_x_ptr;
    int row_start = blockIdx.x * BLOCK_M;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    __shared__ float partial[BLOCK_M][WARPS_PER_BLOCK];

    // Each warp processes a slice of N. Each lane within the warp processes
    // a stride of WARP_SIZE * 2 (since each packed byte = 2 weights).
    int n_per_warp = N / WARPS_PER_BLOCK;
    int n_warp_start = warpId * n_per_warp;

    // Per-row partial sums for this thread
    float local_acc[BLOCK_M];
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) local_acc[i] = 0.0f;

    // half2-vectorized inner loop: load 2 fp16 x values, do 2 mul + 1 add per code-pair
    for (int n = n_warp_start + laneId * 2; n < n_warp_start + n_per_warp; n += WARP_SIZE * 2) {
        // Load x[n], x[n+1] as a fp16x2
        __half2 x_pair = *reinterpret_cast<const __half2*>(&x[n]);

        #pragma unroll
        for (int row_off = 0; row_off < BLOCK_M; row_off++) {
            int row = row_start + row_off;
            if (row >= M) continue;
            uint8_t pbyte = packed[(size_t)row * (N / 2) + (n / 2)];
            // Build (code_low, code_high) as fp16x2
            __half2 w_pair = __floats2half2_rn(
                (float)(pbyte & 0xF),
                (float)((pbyte >> 4) & 0xF));
            // Half2 multiply (1 instruction for 2 fp16 muls), then horizontal add
            __half2 prod = __hmul2(w_pair, x_pair);
            float p = __half2float(__low2half(prod)) + __half2float(__high2half(prod));
            local_acc[row_off] += p;
        }
    }

    // Lane reduction: sum across lanes within the warp for each row.
    #pragma unroll
    for (int row_off = 0; row_off < BLOCK_M; row_off++) {
        float v = local_acc[row_off];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        }
        if (laneId == 0) {
            partial[row_off][warpId] = v;
        }
    }
    __syncthreads();

    // Final reduction (warp 0): sum across warps for each row, apply scale and zero correction.
    if (warpId == 0 && laneId < BLOCK_M) {
        int row_off = laneId;
        int row = row_start + row_off;
        if (row < M) {
            float dot = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                dot += partial[row_off][w];
            }
            // Zero-point correction: y = scale * (dot - zero * sum_x), sum_x precomputed by caller.
            float scale_f = __half2float(scale[row]);
            float zero_f  = __half2float(zero[row]);
            y[row] = __float2half(scale_f * (dot - zero_f * sum_x));
        }
    }
}


__global__ void matvec_int2_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scale,
    const __half* __restrict__ zero,
    const float* __restrict__ sum_x_ptr,
    __half* __restrict__ y,
    int M, int N
) {
    float sum_x = *sum_x_ptr;
    int row_start = blockIdx.x * BLOCK_M;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    __shared__ float partial[BLOCK_M][WARPS_PER_BLOCK];

    int n_per_warp = N / WARPS_PER_BLOCK;
    int n_warp_start = warpId * n_per_warp;

    float local_acc[BLOCK_M];
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) local_acc[i] = 0.0f;

    // half2-vectorized: load x as two half2's (4 fp16), do 2 half2 muls
    for (int n = n_warp_start + laneId * 4; n < n_warp_start + n_per_warp; n += WARP_SIZE * 4) {
        __half2 x01 = *reinterpret_cast<const __half2*>(&x[n]);     // (x[n], x[n+1])
        __half2 x23 = *reinterpret_cast<const __half2*>(&x[n + 2]); // (x[n+2], x[n+3])

        #pragma unroll
        for (int row_off = 0; row_off < BLOCK_M; row_off++) {
            int row = row_start + row_off;
            if (row >= M) continue;
            uint8_t pbyte = packed[(size_t)row * (N / 4) + (n / 4)];
            __half2 w01 = __floats2half2_rn((float)(pbyte & 0x3), (float)((pbyte >> 2) & 0x3));
            __half2 w23 = __floats2half2_rn((float)((pbyte >> 4) & 0x3), (float)((pbyte >> 6) & 0x3));
            __half2 p01 = __hmul2(w01, x01);
            __half2 p23 = __hmul2(w23, x23);
            float s = __half2float(__low2half(p01)) + __half2float(__high2half(p01))
                    + __half2float(__low2half(p23)) + __half2float(__high2half(p23));
            local_acc[row_off] += s;
        }
    }

    #pragma unroll
    for (int row_off = 0; row_off < BLOCK_M; row_off++) {
        float v = local_acc[row_off];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        }
        if (laneId == 0) {
            partial[row_off][warpId] = v;
        }
    }
    __syncthreads();

    if (warpId == 0 && laneId < BLOCK_M) {
        int row_off = laneId;
        int row = row_start + row_off;
        if (row < M) {
            float dot = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                dot += partial[row_off][w];
            }
            float scale_f = __half2float(scale[row]);
            float zero_f  = __half2float(zero[row]);
            y[row] = __float2half(scale_f * (dot - zero_f * sum_x));
        }
    }
}


// ===========================================================================
// Per-GROUP v1 (block-of-rows) fallback kernels — used when N is not divisible
// by 32 (int4) / 64 (int2), so the warp-per-row group kernels can't run.
// Dequant inside the loop with per-group scale/zero; no sum_x argument.
//   y[i] = Σ_j scale[i, gid[j]] * (code[i,j] - zero[i, gid[j]]) * x[j]
// Each warp covers a slice of N; lane reduction then cross-warp reduction.
// ===========================================================================
__global__ void matvec_int4_group_v1_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scale,   // (M, n_groups)
    const __half* __restrict__ zero,    // (M, n_groups)
    const int* __restrict__ gid,        // (N,)
    __half* __restrict__ y,
    int M, int N, int n_groups
) {
    int row_start = blockIdx.x * BLOCK_M;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    __shared__ float partial[BLOCK_M][WARPS_PER_BLOCK];
    int n_per_warp = N / WARPS_PER_BLOCK;
    int n_warp_start = warpId * n_per_warp;
    float local_acc[BLOCK_M];
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) local_acc[i] = 0.0f;
    for (int n = n_warp_start + laneId * 2; n < n_warp_start + n_per_warp; n += WARP_SIZE * 2) {
        __half2 x_pair = *reinterpret_cast<const __half2*>(&x[n]);
        float xl = __half2float(__low2half(x_pair)), xh = __half2float(__high2half(x_pair));
        int g0 = gid[n], g1 = gid[n + 1];
        #pragma unroll
        for (int row_off = 0; row_off < BLOCK_M; row_off++) {
            int row = row_start + row_off;
            if (row >= M) continue;
            uint8_t pbyte = packed[(size_t)row * (N / 2) + (n / 2)];
            const __half* srow = scale + (size_t)row * n_groups;
            const __half* zrow = zero  + (size_t)row * n_groups;
            float s0 = __half2float(srow[g0]), z0 = __half2float(zrow[g0]);
            float s1 = __half2float(srow[g1]), z1 = __half2float(zrow[g1]);
            local_acc[row_off] += s0 * ((float)(pbyte & 0xF) - z0) * xl
                                + s1 * ((float)((pbyte >> 4) & 0xF) - z1) * xh;
        }
    }
    #pragma unroll
    for (int row_off = 0; row_off < BLOCK_M; row_off++) {
        float v = local_acc[row_off];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        if (laneId == 0) partial[row_off][warpId] = v;
    }
    __syncthreads();
    if (warpId == 0 && laneId < BLOCK_M) {
        int row_off = laneId, row = row_start + row_off;
        if (row < M) {
            float dot = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) dot += partial[row_off][w];
            y[row] = __float2half(dot);
        }
    }
}

__global__ void matvec_int2_group_v1_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scale,   // (M, n_groups)
    const __half* __restrict__ zero,    // (M, n_groups)
    const int* __restrict__ gid,        // (N,)
    __half* __restrict__ y,
    int M, int N, int n_groups
) {
    int row_start = blockIdx.x * BLOCK_M;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    __shared__ float partial[BLOCK_M][WARPS_PER_BLOCK];
    int n_per_warp = N / WARPS_PER_BLOCK;
    int n_warp_start = warpId * n_per_warp;
    float local_acc[BLOCK_M];
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) local_acc[i] = 0.0f;
    for (int n = n_warp_start + laneId * 4; n < n_warp_start + n_per_warp; n += WARP_SIZE * 4) {
        __half2 x01 = *reinterpret_cast<const __half2*>(&x[n]);
        __half2 x23 = *reinterpret_cast<const __half2*>(&x[n + 2]);
        float a0 = __half2float(__low2half(x01)), a1 = __half2float(__high2half(x01));
        float a2 = __half2float(__low2half(x23)), a3 = __half2float(__high2half(x23));
        int g0 = gid[n], g1 = gid[n + 1], g2 = gid[n + 2], g3 = gid[n + 3];
        #pragma unroll
        for (int row_off = 0; row_off < BLOCK_M; row_off++) {
            int row = row_start + row_off;
            if (row >= M) continue;
            uint8_t pbyte = packed[(size_t)row * (N / 4) + (n / 4)];
            const __half* srow = scale + (size_t)row * n_groups;
            const __half* zrow = zero  + (size_t)row * n_groups;
            local_acc[row_off] +=
                  __half2float(srow[g0]) * ((float)(pbyte & 0x3)        - __half2float(zrow[g0])) * a0
                + __half2float(srow[g1]) * ((float)((pbyte >> 2) & 0x3) - __half2float(zrow[g1])) * a1
                + __half2float(srow[g2]) * ((float)((pbyte >> 4) & 0x3) - __half2float(zrow[g2])) * a2
                + __half2float(srow[g3]) * ((float)((pbyte >> 6) & 0x3) - __half2float(zrow[g3])) * a3;
        }
    }
    #pragma unroll
    for (int row_off = 0; row_off < BLOCK_M; row_off++) {
        float v = local_acc[row_off];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
        if (laneId == 0) partial[row_off][warpId] = v;
    }
    __syncthreads();
    if (warpId == 0 && laneId < BLOCK_M) {
        int row_off = laneId, row = row_start + row_off;
        if (row < M) {
            float dot = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) dot += partial[row_off][w];
            y[row] = __float2half(dot);
        }
    }
}


// ===========================================================================
// Warp-per-row matvec (the default path, used by the BiIPLinear decode forward):
//   - block = VROWS warps = VROWS output rows (1 warp owns a full row).
//   - no shared-mem cross-warp reduction (each warp reduces its own row).
//   - weights loaded as uint4 (16 B = 32 int4 codes / 64 int2 codes), coalesced
//     across the warp's 32 lanes (512 contiguous bytes per step).
//   - sum_x for the zero-point term is accumulated INSIDE the kernel (v3),
//     removing the separate reduction kernel from the Python forward.
// Assumes N % 32 == 0 (int4) / N % 64 == 0 (int2) — true for Llama dims.
// ===========================================================================
#define VROWS 8

// ===========================================================================
// v3 matvec: warp-per-row, uint4 loads, sum_x computed INTERNALLY (each warp
// already reads all of x for its row), eliminating the separate
// z.float().sum() reduction kernel (one per Linear) in the Python forward.
// No sum_x argument. y[i] = scale[i]*(dot_i - zero[i]*sum_x).
// ===========================================================================
__global__ void matvec_int4_v3_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scale,
    const __half* __restrict__ zero,
    __half* __restrict__ y,
    int M, int N
) {
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * VROWS + warp;
    if (row >= M) return;
    const uint4* wrow4 = reinterpret_cast<const uint4*>(packed + (size_t)row * (N / 2));
    int n_uint4 = N / 32;
    float acc = 0.0f, sx = 0.0f;
    for (int u = lane; u < n_uint4; u += WARP_SIZE) {
        uint4 wv = wrow4[u];
        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wv);
        const __half2* xp = reinterpret_cast<const __half2*>(&x[u * 32]);
        #pragma unroll
        for (int b = 0; b < 16; b++) {
            __half2 xpair = xp[b];
            float xl = __half2float(__low2half(xpair)), xh = __half2float(__high2half(xpair));
            sx += xl + xh;
            uint8_t pb = wb[b];
            acc += (float)(pb & 0xF) * xl + (float)((pb >> 4) & 0xF) * xh;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
        sx  += __shfl_xor_sync(0xFFFFFFFF, sx, off);
    }
    if (lane == 0) {
        float s = __half2float(scale[row]), z = __half2float(zero[row]);
        y[row] = __float2half(s * (acc - z * sx));
    }
}

__global__ void matvec_int2_v3_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scale,
    const __half* __restrict__ zero,
    __half* __restrict__ y,
    int M, int N
) {
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * VROWS + warp;
    if (row >= M) return;
    const uint4* wrow4 = reinterpret_cast<const uint4*>(packed + (size_t)row * (N / 4));
    int n_uint4 = N / 64;
    float acc = 0.0f, sx = 0.0f;
    for (int u = lane; u < n_uint4; u += WARP_SIZE) {
        uint4 wv = wrow4[u];
        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wv);
        const __half2* xp = reinterpret_cast<const __half2*>(&x[u * 64]);
        #pragma unroll
        for (int b = 0; b < 16; b++) {
            __half2 x01 = xp[2 * b], x23 = xp[2 * b + 1];
            float a0 = __half2float(__low2half(x01)), a1 = __half2float(__high2half(x01));
            float a2 = __half2float(__low2half(x23)), a3 = __half2float(__high2half(x23));
            sx += a0 + a1 + a2 + a3;
            uint8_t pb = wb[b];
            acc += (float)(pb & 0x3) * a0 + (float)((pb >> 2) & 0x3) * a1
                 + (float)((pb >> 4) & 0x3) * a2 + (float)((pb >> 6) & 0x3) * a3;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
        sx  += __shfl_xor_sync(0xFFFFFFFF, sx, off);
    }
    if (lane == 0) {
        float s = __half2float(scale[row]), z = __half2float(zero[row]);
        y[row] = __float2half(s * (acc - z * sx));
    }
}


// ===========================================================================
// Per-GROUP (g128) matvec kernels. scale/zero are (M, n_groups); gid[col]
// gives the group of input column col (handles non-contiguous groups from
// act_order). Dequant happens INSIDE the loop:
//   y[i] = Σ_j scale[i, gid[j]] * (code[i,j] - zero[i, gid[j]]) * x[j]
// Warp-per-row layout, same coalesced uint4 loads as v3. No factoring of
// scale/zero out of the sum (they vary per group), so no internal sum_x.
// ===========================================================================
__global__ void matvec_int4_group_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scale,   // (M, n_groups)
    const __half* __restrict__ zero,    // (M, n_groups)
    const int* __restrict__ gid,        // (N,) column -> group
    __half* __restrict__ y,
    int M, int N, int n_groups
) {
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * VROWS + warp;
    if (row >= M) return;
    const uint4* wrow4 = reinterpret_cast<const uint4*>(packed + (size_t)row * (N / 2));
    const __half* srow = scale + (size_t)row * n_groups;
    const __half* zrow = zero  + (size_t)row * n_groups;
    int n_uint4 = N / 32;
    float acc = 0.0f;
    for (int u = lane; u < n_uint4; u += WARP_SIZE) {
        uint4 wv = wrow4[u];
        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wv);
        const __half2* xp = reinterpret_cast<const __half2*>(&x[u * 32]);
        int col0 = u * 32;
        #pragma unroll
        for (int b = 0; b < 16; b++) {
            __half2 xpair = xp[b];
            float xl = __half2float(__low2half(xpair)), xh = __half2float(__high2half(xpair));
            uint8_t pb = wb[b];
            int c0 = col0 + 2 * b;
            int g0 = gid[c0], g1 = gid[c0 + 1];
            float s0 = __half2float(srow[g0]), z0 = __half2float(zrow[g0]);
            float s1 = __half2float(srow[g1]), z1 = __half2float(zrow[g1]);
            acc += s0 * ((float)(pb & 0xF) - z0) * xl
                 + s1 * ((float)((pb >> 4) & 0xF) - z1) * xh;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    }
    if (lane == 0) y[row] = __float2half(acc);
}

__global__ void matvec_int2_group_kernel(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const __half* __restrict__ scale,   // (M, n_groups)
    const __half* __restrict__ zero,    // (M, n_groups)
    const int* __restrict__ gid,        // (N,) column -> group
    __half* __restrict__ y,
    int M, int N, int n_groups
) {
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * VROWS + warp;
    if (row >= M) return;
    const uint4* wrow4 = reinterpret_cast<const uint4*>(packed + (size_t)row * (N / 4));
    const __half* srow = scale + (size_t)row * n_groups;
    const __half* zrow = zero  + (size_t)row * n_groups;
    int n_uint4 = N / 64;
    float acc = 0.0f;
    for (int u = lane; u < n_uint4; u += WARP_SIZE) {
        uint4 wv = wrow4[u];
        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wv);
        const __half2* xp = reinterpret_cast<const __half2*>(&x[u * 64]);
        int col0 = u * 64;
        #pragma unroll
        for (int b = 0; b < 16; b++) {
            __half2 x01 = xp[2 * b], x23 = xp[2 * b + 1];
            float a0 = __half2float(__low2half(x01)), a1 = __half2float(__high2half(x01));
            float a2 = __half2float(__low2half(x23)), a3 = __half2float(__high2half(x23));
            uint8_t pb = wb[b];
            int c0 = col0 + 4 * b;
            int g0 = gid[c0], g1 = gid[c0 + 1], g2 = gid[c0 + 2], g3 = gid[c0 + 3];
            acc += __half2float(srow[g0]) * ((float)(pb & 0x3)        - __half2float(zrow[g0])) * a0
                 + __half2float(srow[g1]) * ((float)((pb >> 2) & 0x3) - __half2float(zrow[g1])) * a1
                 + __half2float(srow[g2]) * ((float)((pb >> 4) & 0x3) - __half2float(zrow[g2])) * a2
                 + __half2float(srow[g3]) * ((float)((pb >> 6) & 0x3) - __half2float(zrow[g3])) * a3;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    }
    if (lane == 0) y[row] = __float2half(acc);
}

torch::Tensor matvec_int4_v3(torch::Tensor x, torch::Tensor packed,
                             torch::Tensor scale, torch::Tensor zero,
                             c10::optional<torch::Tensor> gid = c10::nullopt) {
    int M = scale.size(0), N = x.size(0);
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous(), "matvec_int4_v3: x/packed must be contiguous");
    TORCH_CHECK(N % 32 == 0, "matvec_int4_v3 needs N % 32 == 0, got N=", N);
    auto y = torch::empty({M}, x.options());
    int grid = (M + VROWS - 1) / VROWS;
    if (scale.dim() == 2 && scale.size(1) > 1) {
        // per-group path
        TORCH_CHECK(gid.has_value(), "matvec_int4_v3: per-group scale requires gid");
        int n_groups = scale.size(1);
        auto gid_t = gid.value();
        TORCH_CHECK(gid_t.is_contiguous() && gid_t.numel() == N, "matvec_int4_v3: gid must be contiguous length N");
        TORCH_CHECK(scale.is_contiguous() && zero.is_contiguous(), "matvec_int4_v3: scale/zero must be contiguous");
        matvec_int4_group_kernel<<<grid, BLOCK_THREADS>>>(
            (const __half*)x.data_ptr(), packed.data_ptr<uint8_t>(),
            (const __half*)scale.data_ptr(), (const __half*)zero.data_ptr(),
            gid_t.data_ptr<int>(), (__half*)y.data_ptr(), M, N, n_groups);
        return y;
    }
    matvec_int4_v3_kernel<<<grid, BLOCK_THREADS>>>(
        (const __half*)x.data_ptr(), packed.data_ptr<uint8_t>(),
        (const __half*)scale.data_ptr(), (const __half*)zero.data_ptr(),
        (__half*)y.data_ptr(), M, N);
    return y;
}

torch::Tensor matvec_int2_v3(torch::Tensor x, torch::Tensor packed,
                             torch::Tensor scale, torch::Tensor zero,
                             c10::optional<torch::Tensor> gid = c10::nullopt) {
    int M = scale.size(0), N = x.size(0);
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous(), "matvec_int2_v3: x/packed must be contiguous");
    TORCH_CHECK(N % 64 == 0, "matvec_int2_v3 needs N % 64 == 0, got N=", N);
    auto y = torch::empty({M}, x.options());
    int grid = (M + VROWS - 1) / VROWS;
    if (scale.dim() == 2 && scale.size(1) > 1) {
        TORCH_CHECK(gid.has_value(), "matvec_int2_v3: per-group scale requires gid");
        int n_groups = scale.size(1);
        auto gid_t = gid.value();
        TORCH_CHECK(gid_t.is_contiguous() && gid_t.numel() == N, "matvec_int2_v3: gid must be contiguous length N");
        TORCH_CHECK(scale.is_contiguous() && zero.is_contiguous(), "matvec_int2_v3: scale/zero must be contiguous");
        matvec_int2_group_kernel<<<grid, BLOCK_THREADS>>>(
            (const __half*)x.data_ptr(), packed.data_ptr<uint8_t>(),
            (const __half*)scale.data_ptr(), (const __half*)zero.data_ptr(),
            gid_t.data_ptr<int>(), (__half*)y.data_ptr(), M, N, n_groups);
        return y;
    }
    matvec_int2_v3_kernel<<<grid, BLOCK_THREADS>>>(
        (const __half*)x.data_ptr(), packed.data_ptr<uint8_t>(),
        (const __half*)scale.data_ptr(), (const __half*)zero.data_ptr(),
        (__half*)y.data_ptr(), M, N);
    return y;
}


// ---------------------------------------------------------------------------
// Fused elementwise kernels for BiIP input/output stages.
//
// input_cast_scale:  z_f32[i] = (float)x_f16[i] * mul_f32[i]
//   Fuses the initial fp16->fp32 cast and the input_mul (stage 1 of
//   BiIPLinear.forward, biip_linear.py) into ONE kernel instead of:
//   cast x (1 kernel) + cast mul (1 kernel) + mul (1 kernel).
//
// output_scale_cast: out_f16[i] = (half)(y_f32[i] * mul_f32[i])
//   Fuses the output_mul (stage 5 of BiIPLinear.forward, biip_linear.py) and
//   the final fp32->fp16 cast into ONE kernel instead of: cast mul + mul + cast back.
// ---------------------------------------------------------------------------

__global__ void input_cast_scale_kernel(
    const __half* __restrict__ x,
    const float* __restrict__ mul,
    float* __restrict__ out,
    int N
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        out[i] = __half2float(x[i]) * mul[i];
    }
}

__global__ void output_scale_cast_kernel(
    const float* __restrict__ y,
    const float* __restrict__ mul,
    __half* __restrict__ out,
    int N
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        out[i] = __float2half(y[i] * mul[i]);
    }
}

torch::Tensor input_cast_scale(torch::Tensor x, torch::Tensor mul) {
    int N = x.numel();
    auto out = torch::empty({N}, x.options().dtype(torch::kFloat32));
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 256) blocks = 256;  // grid-stride; cap for small launch
    input_cast_scale_kernel<<<blocks, threads>>>(
        (const __half*)x.data_ptr(), mul.data_ptr<float>(),
        out.data_ptr<float>(), N);
    return out;
}

torch::Tensor output_scale_cast(torch::Tensor y, torch::Tensor mul) {
    int N = y.numel();
    auto out = torch::empty({N}, y.options().dtype(torch::kFloat16));
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 256) blocks = 256;
    output_scale_cast_kernel<<<blocks, threads>>>(
        y.data_ptr<float>(), mul.data_ptr<float>(),
        (__half*)out.data_ptr(), N);
    return out;
}


// Host-side wrappers (PyTorch tensors)
torch::Tensor matvec_int4(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor scale,
    torch::Tensor zero,
    c10::optional<torch::Tensor> sum_x = c10::nullopt,  // (1,) fp32, per-row path only
    c10::optional<torch::Tensor> gid = c10::nullopt     // (N,) int32, per-group path
) {
    int M = scale.size(0);
    int N = x.size(0);
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous(), "matvec_int4: x/packed must be contiguous");
    TORCH_CHECK(N % 16 == 0, "matvec_int4 needs N % 16 == 0, got N=", N);
    auto y = torch::empty({M}, x.options());
    int grid = (M + BLOCK_M - 1) / BLOCK_M;
    if (scale.dim() == 2 && scale.size(1) > 1) {
        TORCH_CHECK(gid.has_value(), "matvec_int4: per-group scale requires gid");
        int n_groups = scale.size(1);
        auto gid_t = gid.value();
        TORCH_CHECK(gid_t.is_contiguous() && gid_t.numel() == N, "matvec_int4: gid must be contiguous length N");
        TORCH_CHECK(scale.is_contiguous() && zero.is_contiguous(), "matvec_int4: scale/zero must be contiguous");
        matvec_int4_group_v1_kernel<<<grid, BLOCK_THREADS>>>(
            (const __half*)x.data_ptr(), packed.data_ptr<uint8_t>(),
            (const __half*)scale.data_ptr(), (const __half*)zero.data_ptr(),
            gid_t.data_ptr<int>(), (__half*)y.data_ptr(), M, N, n_groups);
        return y;
    }
    TORCH_CHECK(sum_x.has_value(), "matvec_int4: per-row scale requires sum_x");
    matvec_int4_kernel<<<grid, BLOCK_THREADS>>>(
        (const __half*)x.data_ptr(),
        packed.data_ptr<uint8_t>(),
        (const __half*)scale.data_ptr(),
        (const __half*)zero.data_ptr(),
        sum_x.value().data_ptr<float>(),
        (__half*)y.data_ptr(),
        M, N
    );
    return y;
}

torch::Tensor matvec_int2(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor scale,
    torch::Tensor zero,
    c10::optional<torch::Tensor> sum_x = c10::nullopt,
    c10::optional<torch::Tensor> gid = c10::nullopt
) {
    int M = scale.size(0);
    int N = x.size(0);
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous(), "matvec_int2: x/packed must be contiguous");
    TORCH_CHECK(N % 16 == 0, "matvec_int2 needs N % 16 == 0, got N=", N);
    auto y = torch::empty({M}, x.options());
    int grid = (M + BLOCK_M - 1) / BLOCK_M;
    if (scale.dim() == 2 && scale.size(1) > 1) {
        TORCH_CHECK(gid.has_value(), "matvec_int2: per-group scale requires gid");
        int n_groups = scale.size(1);
        auto gid_t = gid.value();
        TORCH_CHECK(gid_t.is_contiguous() && gid_t.numel() == N, "matvec_int2: gid must be contiguous length N");
        TORCH_CHECK(scale.is_contiguous() && zero.is_contiguous(), "matvec_int2: scale/zero must be contiguous");
        matvec_int2_group_v1_kernel<<<grid, BLOCK_THREADS>>>(
            (const __half*)x.data_ptr(), packed.data_ptr<uint8_t>(),
            (const __half*)scale.data_ptr(), (const __half*)zero.data_ptr(),
            gid_t.data_ptr<int>(), (__half*)y.data_ptr(), M, N, n_groups);
        return y;
    }
    TORCH_CHECK(sum_x.has_value(), "matvec_int2: per-row scale requires sum_x");
    matvec_int2_kernel<<<grid, BLOCK_THREADS>>>(
        (const __half*)x.data_ptr(),
        packed.data_ptr<uint8_t>(),
        (const __half*)scale.data_ptr(),
        (const __half*)zero.data_ptr(),
        sum_x.value().data_ptr<float>(),
        (__half*)y.data_ptr(),
        M, N
    );
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matvec_int4", &matvec_int4, "Fused dequant+matvec int4",
          py::arg("x"), py::arg("packed"), py::arg("scale"), py::arg("zero"),
          py::arg("sum_x") = c10::nullopt, py::arg("gid") = c10::nullopt);
    m.def("matvec_int2", &matvec_int2, "Fused dequant+matvec int2",
          py::arg("x"), py::arg("packed"), py::arg("scale"), py::arg("zero"),
          py::arg("sum_x") = c10::nullopt, py::arg("gid") = c10::nullopt);
    m.def("input_cast_scale", &input_cast_scale, "Fused fp16->fp32 cast * input_mul");
    m.def("output_scale_cast", &output_scale_cast, "Fused y*output_mul -> fp16");
    m.def("matvec_int4_v3", &matvec_int4_v3, "warp-per-row vectorized int4 matvec, internal sum_x (default)",
          py::arg("x"), py::arg("packed"), py::arg("scale"), py::arg("zero"),
          py::arg("gid") = c10::nullopt);
    m.def("matvec_int2_v3", &matvec_int2_v3, "warp-per-row vectorized int2 matvec, internal sum_x (default)",
          py::arg("x"), py::arg("packed"), py::arg("scale"), py::arg("zero"),
          py::arg("gid") = c10::nullopt);
}
