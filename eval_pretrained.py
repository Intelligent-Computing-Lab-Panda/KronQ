# --- repo path bootstrap: inference-only entry point: needs only common/ + runtime/ (no calibration code) ---
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not all(_os.path.isdir(_os.path.join(_d, s)) for s in ('common', 'runtime')):
    _d = _os.path.dirname(_d)
for _s in ('common', 'runtime'):
    _p = _os.path.join(_d, _s)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
"""Evaluate a PACKED KronQ checkpoint downloaded from the HuggingFace Hub (or a
local packed dir) — WikiText-2 PPL and/or zero-shot. No calibration code needed.

  python eval_pretrained.py meta-llama/Llama-2-7b-hf donghyunli/Llama-2-7b-KronQ-W4A16 --ppl --zs
  python eval_pretrained.py meta-llama/Llama-2-7b-hf ./w4_packed --ppl
"""
import argparse
import model_utils, quant_utils, data_utils, eval_utils, utils
import packed_io

_ZS_DEFAULT = ['piqa', 'hellaswag', 'arc_easy', 'arc_challenge', 'winogrande', 'boolq']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', help='HF model id (architecture + tokenizer), e.g. meta-llama/Llama-2-7b-hf')
    ap.add_argument('packed', help='local packed dir OR HF repo id of packed KronQ weights')
    ap.add_argument('--ppl', action='store_true', help='WikiText-2 perplexity')
    ap.add_argument('--zs', action='store_true', help='zero-shot accuracy (lm-eval)')
    ap.add_argument('--tasks', nargs='+', default=_ZS_DEFAULT)
    ap.add_argument('--seqlen', type=int, default=2048)
    ap.add_argument('--zs_batch_size', type=int, default=16)
    ap.add_argument('--distribute', action='store_true', help='spread model across all visible GPUs')
    args = ap.parse_args()
    hf = _os.environ.get('HF_TOKEN')

    packed_dir = args.packed
    if not _os.path.isdir(packed_dir):
        from huggingface_hub import snapshot_download
        print(f"Downloading packed weights from HF: {args.packed}")
        packed_dir = snapshot_download(args.packed, repo_type='model', token=hf)

    model = model_utils.get_model(args.model, hf)
    model.eval()
    quant_utils.add_actquant(model)
    n = packed_io.load_packed(model, packed_dir)
    print(f"Loaded {n} packed BiIP linears from {packed_dir}", flush=True)
    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)

    if args.ppl:
        model.seqlen = args.seqlen
        ev = argparse.Namespace(bsz=1, capture_layer_io=False, layer_idx=-1, eval_dataset='wikitext2')
        testloader = data_utils.get_loaders('wikitext2', seed=0, model=args.model,
                                            seqlen=args.seqlen, hf_token=hf, eval_mode=True)
        ppl = eval_utils.evaluator(model, testloader, utils.DEV, ev)
        print(f"WikiText-2 PPL: {ppl:.3f}", flush=True)

    if args.zs:
        if args.ppl:  # PPL eval offloads decoder layers to CPU; restore to GPU
            if args.distribute:
                utils.distribute_model(model)
            else:
                model.to(utils.DEV)
        import inspect, lm_eval
        from lm_eval.models.huggingface import HFLM
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, token=hf)
        hflm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.zs_batch_size)
        kw = {'tasks': args.tasks, 'batch_size': args.zs_batch_size}
        accepted = inspect.signature(lm_eval.simple_evaluate).parameters
        kw = {k: v for k, v in kw.items() if k in accepted}
        results = lm_eval.simple_evaluate(hflm, **kw)['results']

        def pick(r):
            for k in ('acc_norm,none', 'acc,none'):
                if k in r:
                    return r[k]
            return next((v for v in r.values() if isinstance(v, (int, float))), float('nan'))
        vals = {t: round(pick(r), 4) for t, r in results.items()}
        vals['acc_avg'] = round(sum(vals.values()) / len(vals), 4)
        print(vals, flush=True)


if __name__ == '__main__':
    main()
