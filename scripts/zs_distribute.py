# --- repo path bootstrap (walks up to repo root) ---
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not all(_os.path.isdir(_os.path.join(_d, s)) for s in ('common', 'runtime', 'calib')):
    _d = _os.path.dirname(_d)
for _s in ('common', 'runtime', 'calib'):
    _p = _os.path.join(_d, _s)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
"""Zero-shot for a STANDARD fp16 model too large for one GPU (e.g. a 70B fp16
model) — loads with device_map across all visible GPUs, runs the 6-task suite.

  python scripts/zs_distribute.py <model_dir_or_id> [base_model_id_for_tokenizer]
"""
import inspect, torch, transformers
import lm_eval
from lm_eval.models.huggingface import HFLM

src = _sys.argv[1]
tok_src = _sys.argv[2] if len(_sys.argv) > 2 else src
hf = _os.environ.get('HF_TOKEN')
TASKS = ['piqa', 'hellaswag', 'arc_easy', 'arc_challenge', 'winogrande', 'boolq']

model = transformers.AutoModelForCausalLM.from_pretrained(
    src, torch_dtype=torch.float16, device_map="auto", token=hf)
model.eval()
tok = transformers.AutoTokenizer.from_pretrained(tok_src, use_fast=False, token=hf)
hflm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
kw = {'tasks': TASKS, 'batch_size': 16}
kw = {k: v for k, v in kw.items() if k in inspect.signature(lm_eval.simple_evaluate).parameters}
results = lm_eval.simple_evaluate(hflm, **kw)['results']

def pick(r):
    for k in ('acc_norm,none', 'acc,none'):
        if k in r:
            return r[k]
    return next((v for v in r.values() if isinstance(v, (int, float))), float('nan'))
vals = {t: round(pick(r), 4) for t, r in results.items()}
vals['acc_avg'] = round(sum(vals.values()) / len(vals), 4)
print(vals, flush=True)
