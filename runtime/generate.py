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
"""Generate text from a KronQ deploy checkpoint (any Llama model).

Loads an fp16 deploy `.pth` (saved by main.py --save_qmodel_path) and
compresses the weights in-memory to packed int4/int2 via
compress_model_to_lowbit; the matvec kernel then unpacks them on the fly
(int-weight x fp16-activation). Prefill uses a dequant path, per-token decode
uses the matvec kernel.

Usage: python generate.py <model> <ckpt.pth> <bits>
  e.g. python generate.py meta-llama/Llama-2-7b-hf kronq_w4_unrotated_had.pth 4
"""
import sys, argparse, torch, transformers
import bench_decode
from lowbit_utils import compress_model_to_lowbit

PROMPTS = [
    "The capital of France is",
    "In a shocking turn of events, scientists discovered that",
    "Here is a simple recipe for chocolate chip cookies:",
]


def main():
    model_name, ckpt, bits = sys.argv[1], sys.argv[2], int(sys.argv[3])
    dev = "cuda"
    args = argparse.Namespace(model=model_name, hf_token=None, seed=0, rotate=False,
                              rotate_mode="hadamard", fp32_had=False,
                              load_qmodel_path=ckpt)
    m = bench_decode.build_model(args).cuda().eval()
    sd = torch.load(ckpt, map_location="cpu")
    n, err = compress_model_to_lowbit(m, bits, w_quantizers=sd.get("w_quantizers"))
    print(f"[{model_name} W{bits}] compressed {n} Linears, recon err {err:.2e}\n", flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
        print("=" * 70)
        print(tok.decode(out[0], skip_special_tokens=True), flush=True)
    print("=" * 70)


if __name__ == "__main__":
    main()
