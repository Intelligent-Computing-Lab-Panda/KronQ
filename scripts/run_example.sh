#!/bin/bash
# End-to-end KronQ example on Llama-2-7B (W4). Run from the repo root.
set -euo pipefail

MODEL=meta-llama/Llama-2-7b-hf
GDIR=grad_cache/llama-2-7b
CKPT=out/kronq_w4.pth
mkdir -p out

# 1) One-time H_G cache (or download the companion HG dataset from the Hub — see README).
python calib/precompute_gradients.py --model "$MODEL" --output "$GDIR" --nsamples 128

# 2) Quantize (fake-quant) + WikiText-2 PPL, and save a deploy ckpt.
python main.py --model "$MODEL" \
    --w_bits 4 --w_groupsize -1 --w_clip --w_asym --a_bits 16 --act_order \
    --bi_calibration --use_gptaq \
    --incoh_rotate --incoh_kernel had --incoh_mode full \
    --alpha 0.25 --grad_dir "$GDIR" \
    --save_unrotated_kronq --save_qmodel_path "$CKPT"

# 3) Real-quant deploy: build the kernel once, then bench + generate.
pip install ./runtime/kronq_kernels
python runtime/bench_decode.py --model "$MODEL" --bench_mode single_tok \
    --use_cuda_graph --simulate_lowbit_storage 4 --load_qmodel_path "$CKPT"
python runtime/generate.py "$MODEL" "$CKPT" 4
