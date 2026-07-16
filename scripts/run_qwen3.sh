#!/bin/bash
# End-to-end KronQ example on Qwen3-8B (W4, weight-only). Run from the repo root.
# Measured on 1x A100 (WikiText-2, seqlen 2048): FP16 9.72, KronQ W4 10.30, GPTAQ W4 10.65.
set -euo pipefail

MODEL=Qwen/Qwen3-8B
GDIR=grad_cache/qwen3-8b

# 1) One-time H_G cache. QK-Norm is detected automatically and q/k gradients are
#    captured at the projection output (see calib/precompute_gradients.py).
python calib/precompute_gradients.py --model "$MODEL" --output "$GDIR" --nsamples 128 --seqlen 2048

# 2) KronQ W4 quantize (fake-quant) + WikiText-2 PPL.
python main.py --model "$MODEL" \
    --w_bits 4 --w_groupsize -1 --w_clip --w_asym --a_bits 16 --act_order \
    --bi_calibration --use_gptaq --incoh_rotate --incoh_kernel had --incoh_mode full \
    --alpha 0.25 --grad_dir "$GDIR"

# Optional ablation: restrict BiIP to a sublayer group (plain GPTAQ elsewhere), e.g.
#   KRONQ_INCOH_GROUP=qk python main.py ...   # groups: qk | no_qk | attn | mlp
