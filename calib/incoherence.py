"""
Incoherence Processing Utilities for KronQ.

Extends QuIP's incoherence processing to the bidirectional (H_X, H_G) setting,
with support for GPTAQ's asymmetric cross-covariance dXXT:
  - dXXT = E[dx * x_tilde^T] transforms identically to H_X on the input side:
      dXXT_rotated = SU @ D2^{-1} @ (dXXT / H_max) @ D2^{-1} @ SU^T
  - dXXT is NOT symmetrized (it's a cross-covariance, not a covariance)

Modes:
  - 'full'              : D1 + D2 rescaling, S_U + S_V rotation
  - 'sv_only'           : D1 rescaling + S_V rotation only
  - 'sv_only_no_rescale': S_V rotation only
  - 'no_rescale'        : S_U + S_V rotation, no rescaling
  - 'su_only'           : input-side only (S_X rescale + S_U rotation)
  - 'rescale_x_only'    : both rotations (S_U + S_V), input-side rescale only (S_X)
  - 'rescale_g_only'    : both rotations (S_U + S_V), output-side rescale only (S_G)
"""

import math
import torch
import logging

import scipy.stats

try:
    import primefac
except ImportError:
    raise ImportError(
        "primefac is required for butterfly orthogonal matrices. "
        "Install with: pip install primefac"
    )

import hadamard_utils

VALID_MODES = {'full', 'sv_only', 'sv_only_no_rescale', 'no_rescale', 'su_only',
               'rescale_x_only', 'rescale_g_only'}
VALID_KERNELS = {'kron', 'had'}


# ============================================================
# Helpers for 'had' kernel (randomized Hadamard transform)
# ============================================================
# Randomized Hadamard transform (QuIP-style). matmul_hadUt for forward,
# matmul_hadU for inverse — correct for non-power-of-2 dims (e.g. 11008 = 172·64)
# where the structured Hadamard is NOT symmetric. For rotation S = D · U^T:
#   forward (RHT): x → S^T x = U D x  (matmul_hadUt(x * sign)); inverse: S x = D U^T x

def _rand_sign(d, dtype, dev):
    return ((torch.randn(d, device=dev).sign() + 1e-5).sign()).to(dtype=dtype)


def _rht_H(H, sign):
    """Forward RHT for symmetric H[n×n]: H ← U^T_n D H D U_n. Sign vector length n."""
    return hadamard_utils.matmul_hadUt(hadamard_utils.matmul_hadUt(H * sign).T * sign)


def _rht_W(W, sign_in, sign_out):
    """Forward RHT for W[m×n]: W ← U^T_m D_out W D_in U_n. Returns same shape."""
    return hadamard_utils.matmul_hadUt(
        hadamard_utils.matmul_hadUt(W.T * sign_out).T * sign_in
    )


def _rht_Q_inverse(Q, sign_in, sign_out):
    """Inverse RHT for Q[m×n] (the quantized W in rotated space).
    Q_orig = D_out U_m Q U^T_n D_in."""
    return (hadamard_utils.matmul_hadU(
        (hadamard_utils.matmul_hadU(Q) * sign_in).T) * sign_out).T


# ============================================================
# Random orthogonal butterfly matrix generation
# ============================================================

def _butterfly_factors(n):
    return list(primefac.primefac(n))


def _gen_rand_orthos(m, p):
    if p != 2:
        return torch.tensor(
            scipy.stats.special_ortho_group.rvs(p, size=m)
        ).to(torch.float32)
    X = torch.zeros(m, 2, 2)
    t = torch.rand(m) * (2 * math.pi)
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    X[:, 0, 0] = cos_t
    X[:, 1, 1] = cos_t
    X[:, 0, 1] = sin_t
    X[:, 1, 0] = -sin_t
    return X


def _gen_rand_ortho_butterfly_noblock(n):
    """Generate random orthogonal butterfly parameters (compact form)."""
    return (
        [_gen_rand_orthos(1, p) for p in _butterfly_factors(n)],
        torch.randperm(n),
        torch.randperm(n),
    )


def _mul_ortho_butterfly(Bpp, x):
    """Multiply vector/matrix x by a random orthogonal butterfly matrix."""
    (B, p_in, p_out) = Bpp
    assert len(x.shape) in (1, 2)
    orig_dim = len(x.shape)
    if orig_dim == 1:
        (n,) = x.shape
        x = x.reshape(n, 1)
    (n, q) = x.shape
    x = x[p_in, :]
    pfn = tuple(_butterfly_factors(n))
    for i in range(len(pfn)):
        mpfx = math.prod(pfn[0:i])
        p = pfn[i]
        msfx = math.prod(pfn[(i + 1):])
        x = (
            x.reshape(mpfx, p, msfx, q)
            .permute(0, 2, 1, 3)
            .reshape(mpfx * msfx, p, q)
        )
        x = B[i] @ x
        x = (
            x.reshape(mpfx, msfx, p, q)
            .permute(0, 2, 1, 3)
            .reshape(n, q)
        )
    x = x[p_out, :]
    if orig_dim == 1:
        x = x.reshape(n)
    return x


def rand_ortho_butterfly_noblock(n):
    """Generate a dense n x n random orthogonal butterfly matrix."""
    return _mul_ortho_butterfly(
        _gen_rand_ortho_butterfly_noblock(n),
        torch.eye(n),
    )


# ============================================================
# KronQ Bidirectional Incoherence Processing
# ============================================================

def incoherence_preprocess_kronq(W, H, G, dXXT, dev, mode='full', kernel='kron'):
    """
    Bidirectional incoherence preprocessing for KronQ.

    Transforms W, H, G, and dXXT to make the Hessian incoherent.
    The KronQ proxy objective is invariant under this transformation:
        L = (1/2) tr(dW^T H_G dW H_X) - tr(dW^T H_G W Delta)

    dXXT (= Delta = E[dx * x_tilde^T]) lives in the input space and
    transforms identically to H on the input side:
        dXXT_rot = SU @ D2^{-1} @ (dXXT / H_max) @ D2^{-1} @ SU^T

    IMPORTANT: dXXT is NOT symmetrized (it's a cross-covariance).

    Args:
        W:     [m, n] weight matrix (float, on device)
        H:     [n, n] activation covariance H_X (float, on device)
        G:     [m, m] gradient covariance H_G (float, on device)
        dXXT:  [n, n] or None - asymmetric cross-covariance from GPTAQ
        dev:   torch device
        mode:  str, one of VALID_MODES

    Returns:
        Wr, Hr, Gr, dXXT_r, SU_cpu, SV_cpu, scaleH_cpu, scaleG_cpu
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}")
    if kernel not in VALID_KERNELS:
        raise ValueError(f"Invalid kernel '{kernel}'. Must be one of {VALID_KERNELS}")

    m, n = W.shape

    do_rescale_input  = mode in ('full', 'su_only', 'rescale_x_only')
    do_rescale_output = mode in ('full', 'sv_only', 'rescale_g_only')
    do_rotate_input   = mode in ('full', 'no_rescale', 'su_only',
                                  'rescale_x_only', 'rescale_g_only')
    do_rotate_output  = mode in ('full', 'sv_only', 'sv_only_no_rescale', 'no_rescale',
                                  'rescale_x_only', 'rescale_g_only')

    scaleH_cpu = None
    scaleG_cpu = None
    SU_cpu = None
    SV_cpu = None

    # ---- Step 1a: Input-side diagonal rescaling (columns) ----
    if do_rescale_input:
        H_max = H.abs().max().clamp(min=1e-12)
        H = H / H_max
        # dXXT gets the SAME H_max normalization
        if dXXT is not None:
            dXXT = dXXT / H_max

        diagH = torch.diag(H).clamp(min=1e-8)
        diagW2_col = torch.diag(W.T @ W).clamp(min=1e-8)
        scaleH = (diagH / diagW2_col).sqrt().sqrt().clamp(min=1e-8)

        W = W * scaleH[None, :]
        H = H / scaleH[None, :] / scaleH[:, None]
        # dXXT gets the SAME D2^{-1} rescaling on both dims
        if dXXT is not None:
            dXXT = dXXT / scaleH[None, :] / scaleH[:, None]

        scaleH_cpu = scaleH.cpu()
        del scaleH
    else:
        H_max = H.abs().max().clamp(min=1e-12)
        H = H / H_max
        # dXXT still needs H_max normalization for P consistency
        if dXXT is not None:
            dXXT = dXXT / H_max

    # ---- Step 1b: Output-side diagonal rescaling (rows) ----
    if do_rescale_output:
        G_max = G.abs().max().clamp(min=1e-12)
        G_norm = G / G_max
        diagG = torch.diag(G_norm).clamp(min=1e-8)
        diagW2_row = torch.diag(W @ W.T).clamp(min=1e-8)
        scaleG = (diagG / diagW2_row).sqrt().sqrt().clamp(min=1e-8)

        W = W * scaleG[:, None]
        G = G_norm / scaleG[None, :] / scaleG[:, None]

        scaleG_cpu = scaleG.cpu()
        del scaleG
    else:
        G_max = G.abs().max().clamp(min=1e-12)
        G = G / G_max

    # ---- Step 2: Random orthogonal rotations ----
    if kernel == 'kron':
        if do_rotate_input:
            SU = rand_ortho_butterfly_noblock(n).to(dtype=W.dtype, device=dev)
            H = SU @ H @ SU.T
            W = W @ SU.T
            if dXXT is not None:
                dXXT = SU @ dXXT @ SU.T
            SU_cpu = SU.cpu()
            del SU
        if do_rotate_output:
            SV = rand_ortho_butterfly_noblock(m).to(dtype=W.dtype, device=dev)
            G = SV @ G @ SV.T
            W = SV @ W
            SV_cpu = SV.cpu()
            del SV
    else:  # 'had' — QuIP-style randomized Hadamard transform
        # Generate sign vectors first since RHT_W needs both
        sign_in_dev = _rand_sign(n, dtype=W.dtype, dev=dev) if do_rotate_input else None
        sign_out_dev = _rand_sign(m, dtype=W.dtype, dev=dev) if do_rotate_output else None
        # _rht_W rotates both sides in one call; an all-ones sign stands in for a
        # disabled side (the Hadamard itself still applies there).
        sign_in_for_W = sign_in_dev if sign_in_dev is not None else torch.ones(n, dtype=W.dtype, device=dev)
        sign_out_for_W = sign_out_dev if sign_out_dev is not None else torch.ones(m, dtype=W.dtype, device=dev)
        if do_rotate_input or do_rotate_output:
            W = _rht_W(W, sign_in_for_W, sign_out_for_W)
        if do_rotate_input:
            H = _rht_H(H, sign_in_dev)
            if dXXT is not None:
                # dXXT is a cross-covariance (NOT symmetric). _rht_H assumes
                # symmetry — would silently transform dXXT^T instead. Use _rht_W
                # which applies S @ M @ S^T for arbitrary square M.
                dXXT = _rht_W(dXXT, sign_in_dev, sign_in_dev)
            SU_cpu = sign_in_dev.cpu()
            del sign_in_dev
        if do_rotate_output:
            G = _rht_H(G, sign_out_dev)
            SV_cpu = sign_out_dev.cpu()
            del sign_out_dev
        del sign_in_for_W, sign_out_for_W

    # Symmetrize H and G only (they are covariances)
    H = (H + H.T) / 2
    G = (G + G.T) / 2
    # dXXT is NOT symmetrized — it's a cross-covariance

    torch.cuda.empty_cache()

    return W, H, G, dXXT, SU_cpu, SV_cpu, scaleH_cpu, scaleG_cpu


def incoherence_postprocess_kronq(Q, SU, SV, scaleH, scaleG, dev, kernel='kron'):
    """
    Reverse incoherence preprocessing.

    Handles any combination of None components (from different modes).

    Args:
        Q:      [m, n] quantized weights in rotated space (on device)
        SU:     for kernel='kron': [n, n] dense; for kernel='had': [n] sign vector
        SV:     for kernel='kron': [m, m] dense; for kernel='had': [m] sign vector
        scaleH: [n] or None (on CPU)
        scaleG: [m] or None (on CPU)
        dev:    torch device
        kernel: 'kron' (dense matrix) or 'had' (sign vector × Hadamard)

    Returns:
        Q in original space (on device)
    """
    Q = Q.float()

    # Undo rotation
    if kernel == 'kron':
        if SV is not None:
            SV = SV.to(dev)
            Q = SV.T @ Q
            del SV
        if SU is not None:
            SU = SU.to(dev)
            Q = Q @ SU
            del SU
    else:  # 'had' — single-call inverse via _rht_Q_inverse
        if SU is not None or SV is not None:
            sign_in_for_Q = SU.to(dev) if SU is not None else torch.ones(Q.shape[1], dtype=Q.dtype, device=dev)
            sign_out_for_Q = SV.to(dev) if SV is not None else torch.ones(Q.shape[0], dtype=Q.dtype, device=dev)
            Q = _rht_Q_inverse(Q, sign_in_for_Q, sign_out_for_Q)
            del sign_in_for_Q, sign_out_for_Q
            if SU is not None: del SU
            if SV is not None: del SV

    # Undo output-side rescaling
    if scaleG is not None:
        scaleG = scaleG.to(dev)
        Q = Q / scaleG[:, None]
        del scaleG

    # Undo input-side rescaling
    if scaleH is not None:
        scaleH = scaleH.to(dev)
        Q = Q / scaleH[None, :]
        del scaleH

    torch.cuda.empty_cache()

    if not torch.all(torch.isfinite(Q)):
        logging.warning("Non-finite values after incoherence postprocessing!")

    return Q

