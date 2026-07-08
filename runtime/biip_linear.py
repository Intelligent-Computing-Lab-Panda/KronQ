"""BiIPLinear runtime forward + load helpers.
Inference-only: applies BiIP transforms from biip_* buffers (no calibration deps).
"""
import os
import torch
import hadamard_utils

# KRONQ_DISABLE_A=1 falls back to the unfused cast+mul path.
_A_FUSION_DISABLED = os.environ.get('KRONQ_DISABLE_A', '0') == '1'
# matvec v3 (warp-per-row, vectorized loads, internal sum_x) is the default; it
# falls back to v1 when dims aren't divisible (N%32 int4 / N%64 int2).
# KRONQ_DISABLE_V3=1 forces the v1 path.
_MATVEC_V3_ON = os.environ.get('KRONQ_DISABLE_V3', '0') != '1'


class BiIPLinear(torch.nn.Linear):
    """A Linear whose forward applies BiIP transforms read from biip_* buffers.

    Activate on an existing Linear via class-swap:
        linear.__class__ = BiIPLinear

    Buffer shapes auto-detected:
      kron mode:  biip_SU/SV are dense d×d matrices
      had mode:   biip_SU/SV are 1D sign vectors

    Forward (QuIP#-style): cast input to fp32, do all rotations/scales/signs in
    fp32, dip to fp16 only for the heavy W matmul, return in the input dtype.
    Hadamards via `matmul_hadU_cuda` with precomputed `had_left_T` (input side)
    and `had_right` (output side).
    """

    def _ensure_had_kernels(self):
        """Lazily precompute the Hadamard kernel buffers on first forward.
        Cast to fp32 + device up-front so subsequent `.to()` calls inside
        matmul_hadU_cuda are no-ops (CUDA-graph capture forbids allocations)."""
        if getattr(self, '_biip_had_initialized', False):
            return
        in_features = self.weight.shape[1]
        out_features = self.weight.shape[0]
        dev = self.weight.device
        had_left, K_left = hadamard_utils.get_hadK(in_features)
        if had_left is not None:
            had_left = had_left.T.contiguous().to(device=dev, dtype=torch.float32)
        self.register_buffer('_had_left_T', had_left, persistent=False)
        had_right, K_right = hadamard_utils.get_hadK(out_features)
        if had_right is not None:
            had_right = had_right.to(device=dev, dtype=torch.float32)
        self.register_buffer('_had_right', had_right, persistent=False)
        self._K_left = K_left
        self._K_right = K_right
        self._biip_had_initialized = True

    def _ensure_fused_mul(self):
        """Lazily cache fp32 copies of biip_input_mul / biip_output_mul so the
        fused input_cast_scale / output_scale_cast elementwise kernels can
        read them directly without a per-call .to(float32) cast. Created on first
        forward (before CUDA-graph capture, so graph-safe)."""
        if getattr(self, '_biip_fused_mul_ready', False):
            return
        im = getattr(self, 'biip_input_mul', None)
        om = getattr(self, 'biip_output_mul', None)
        if im is not None:
            self.register_buffer('_input_mul_f32', im.to(torch.float32).contiguous(),
                                 persistent=False)
        if om is not None:
            self.register_buffer('_output_mul_f32', om.to(torch.float32).contiguous(),
                                 persistent=False)
        self._biip_fused_mul_ready = True

    def forward(self, x):
        in_dtype = x.dtype
        biip_SU = getattr(self, 'biip_SU', None)
        biip_SV = getattr(self, 'biip_SV', None)
        had_mode = (biip_SU is not None and biip_SU.dim() == 1) or \
                   (biip_SV is not None and biip_SV.dim() == 1)
        if had_mode:
            self._ensure_had_kernels()

        # single-token decode → fused elementwise kernels fast path
        bsz_flat = 1
        for d in x.shape[:-1]:
            bsz_flat *= d
        biip_input_mul = getattr(self, 'biip_input_mul', None)
        _KK = None
        use_fused_elem = (bsz_flat == 1 and not torch.is_grad_enabled()
                          and not _A_FUSION_DISABLED)
        if use_fused_elem:
            try:
                import kronq_kernels as _KK
            except ImportError:
                _KK = None

        # 1. Input cast + rescale + sign.
        if (use_fused_elem and _KK is not None and biip_input_mul is not None
                and in_dtype == torch.float16):
            # Fused: z_f32 = (float)x_f16 * input_mul  (one kernel, was cast+cast+mul)
            self._ensure_fused_mul()
            z = _KK.input_cast_scale(x.reshape(-1), self._input_mul_f32).reshape(x.shape)
        else:
            # Cast input to fp32 — all rotation/scale/sign ops in fp32 (per QuIP#).
            z = x.to(torch.float32)
            if biip_input_mul is not None:
                z = z * biip_input_mul.to(torch.float32)
            else:
                scaleH = getattr(self, 'biip_scaleH', None)
                if scaleH is not None:
                    z = z / scaleH.to(torch.float32)
                if biip_SU is not None and biip_SU.dim() == 1:
                    z = z * biip_SU.to(torch.float32)
        # 2. Input rotation (Hadamard, structured)
        if biip_SU is not None:
            if biip_SU.dim() == 1:  # had mode
                z = hadamard_utils.matmul_hadU_cuda(z, self._had_left_T.to(z.device) if self._had_left_T is not None else None, self._K_left)
            else:  # kron mode (dense)
                z = z @ biip_SU.to(torch.float32).T
        # 3. Heavy W matmul in fp16 briefly (per QuIP#)
        if getattr(self, 'biip_w_codes', None) is not None:
            bsz_flat = 1
            for d in z.shape[:-1]:
                bsz_flat *= d
            # Per-group (g128): biip_w_scale is 2D (m, n_groups) + biip_w_gid (n,).
            biip_w_gid = getattr(self, 'biip_w_gid', None)
            is_group = (self.biip_w_scale.dim() == 2 and self.biip_w_scale.shape[1] > 1)
            if bsz_flat == 1:
                # Prefer the raw-CUDA kernel over Triton when available
                z_fp16 = z.to(in_dtype).reshape(-1)
                try:
                    import kronq_kernels
                    n_in = z_fp16.numel()
                    bits = self.biip_w_bits
                    v3_ok = (n_in % 32 == 0) if bits == 4 else (n_in % 64 == 0)
                    if is_group:
                        # Per-group: scale/zero vary per column-group; pass gid.
                        if _MATVEC_V3_ON and v3_ok:
                            fn3 = (kronq_kernels.matvec_int4_v3 if bits == 4
                                   else kronq_kernels.matvec_int2_v3)
                            y_flat = fn3(z_fp16, self.biip_w_codes,
                                         self.biip_w_scale, self.biip_w_zero,
                                         biip_w_gid)
                        else:
                            fn = (kronq_kernels.matvec_int4 if bits == 4
                                  else kronq_kernels.matvec_int2)
                            y_flat = fn(z_fp16, self.biip_w_codes,
                                        self.biip_w_scale, self.biip_w_zero,
                                        None, biip_w_gid)
                    elif _MATVEC_V3_ON and v3_ok:
                        # v3 computes sum_x internally — no separate reduction kernel.
                        fn3 = (kronq_kernels.matvec_int4_v3 if bits == 4
                               else kronq_kernels.matvec_int2_v3)
                        y_flat = fn3(z_fp16, self.biip_w_codes,
                                     self.biip_w_scale, self.biip_w_zero)
                    else:
                        # Keep sum_x on GPU — `.item()` would CPU-sync and break CUDA graph capture
                        sum_x = z_fp16.float().sum().unsqueeze(0)
                        fn = (kronq_kernels.matvec_int4 if bits == 4
                              else kronq_kernels.matvec_int2)
                        y_flat = fn(z_fp16, self.biip_w_codes,
                                    self.biip_w_scale, self.biip_w_zero, sum_x)
                except ImportError:
                    from lowbit_triton import fused_dequant_matvec
                    y_flat = fused_dequant_matvec(
                        z_fp16, self.biip_w_codes,
                        self.biip_w_scale, self.biip_w_zero,
                        self.biip_w_bits, self.biip_w_out_dim,
                    )
                y = y_flat.reshape(*z.shape[:-1], -1).to(torch.float32)
            else:
                if is_group:
                    from lowbit_utils import dequant_packed_group
                    W_dequant = dequant_packed_group(
                        self.biip_w_codes, self.biip_w_scale, self.biip_w_zero,
                        biip_w_gid, self.biip_w_bits, self.biip_w_out_dim,
                    )
                else:
                    from lowbit_utils import dequant_packed
                    W_dequant = dequant_packed(
                        self.biip_w_codes, self.biip_w_scale, self.biip_w_zero,
                        self.biip_w_bits, self.biip_w_out_dim,
                    )
                y = (z.to(in_dtype) @ W_dequant.T).to(torch.float32)
        else:
            y = (z.to(in_dtype) @ self.weight.T).to(torch.float32)
        # 4. Output rotation (Hadamard, structured)
        if biip_SV is not None:
            if biip_SV.dim() == 1:  # had mode
                y = hadamard_utils.matmul_hadU_cuda(y, self._had_right.to(y.device) if self._had_right is not None else None, self._K_right)
            else:  # kron mode (dense)
                y = y @ biip_SV.to(torch.float32)
        # 5. Output rescale + sign (fused). Fast path: fuse rescale + final
        #    fp32->fp16 cast into one kernel when single-token, no bias.
        biip_output_mul = getattr(self, 'biip_output_mul', None)
        if (use_fused_elem and _KK is not None and biip_output_mul is not None
                and self.bias is None and in_dtype == torch.float16):
            self._ensure_fused_mul()
            return _KK.output_scale_cast(y.reshape(-1), self._output_mul_f32).reshape(y.shape)
        if biip_output_mul is not None:
            y = y * biip_output_mul.to(torch.float32)
        else:
            if biip_SV is not None and biip_SV.dim() == 1:
                y = y * biip_SV.to(torch.float32)
            scaleG = getattr(self, 'biip_scaleG', None)
            if scaleG is not None:
                y = y / scaleG.to(torch.float32)
        # 6. Bias
        if self.bias is not None:
            y = y + self.bias.to(torch.float32)
        return y.to(in_dtype)


def register_biip_buffers_from_state_dict(model, state_dict):
    """For every `*.biip_<name>` key in state_dict, register an empty buffer of
    matching shape/dtype on the parent module so that `load_state_dict` can fill
    it in. Without this, `load_state_dict(strict=False)` silently drops biip_*
    entries because they don't exist on freshly built models.
    """
    for key, tensor in state_dict.items():
        if '.biip_' not in key:
            continue
        parent_key, buffer_name = key.rsplit('.', 1)
        module = model
        for part in parent_key.split('.'):
            module = getattr(module, part)
        module.register_buffer(buffer_name, torch.empty_like(tensor))


def _has_any_biip_buffer(linear):
    # SU/SV/scaleH/scaleG identify a fp16 (un-packed) BiIP Linear; biip_w_codes /
    # biip_input_mul identify a PACKED deploy Linear whose SU/SV were already
    # folded into input_mul/output_mul (so they aren't stored separately).
    return any(
        getattr(linear, name, None) is not None
        for name in ('biip_SU', 'biip_SV', 'biip_scaleH', 'biip_scaleG',
                     'biip_w_codes', 'biip_input_mul')
    )


def wrap_linears_with_biip(model):
    """Class-swap every Linear that carries biip_* buffers to BiIPLinear.

    Idempotent — Linears already swapped (class is BiIPLinear) are skipped.
    Works at any granularity: pass the top-level model or a single decoder layer.
    """
    n_wrapped = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and not isinstance(module, BiIPLinear):
            if _has_any_biip_buffer(module):
                module.__class__ = BiIPLinear
                n_wrapped += 1
    return n_wrapped
