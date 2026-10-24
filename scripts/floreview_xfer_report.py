"""AUC (edits vs the clip's own untampered control), clip-cluster bootstrap CI, per-edit AUC and TPR at 10% FPR on the FloreView transfer set."""
import sys, pickle, json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from custody.forge import OPS

rng = np.random.default_rng(0)
base = pickle.load(open(sys.argv[1], "rb")); gbm = pickle.load(open(sys.argv[2], "rb"))
S = {**{k: v for k, v in base.items() if k != "device"}, **{k: v for k, v in gbm.items() if k != "device"}}
n = len(base["device"]); out = {}

def auc_idx(neg, pos, idx):
    p = np.concatenate([pos[o][idx] for o in pos]) if isinstance(pos, dict) else pos[idx]
    ng = neg[idx]
    return roc_auc_score(np.r_[np.zeros(len(ng)), np.ones(len(p))], np.r_[ng, p])

for m, g in S.items():
    neg = np.asarray(g["ctrl"]); pos = {o: np.asarray(g[o]) for o in OPS}
    allidx = np.arange(n)
    pt = auc_idx(neg, pos, allidx)
    bs = [auc_idx(neg, pos, rng.integers(0, n, n)) for _ in range(2000)]
    per = {o: round(float(auc_idx(neg, pos[o], allidx)), 3) for o in OPS}
    y = np.r_[np.zeros(n), np.ones(n * len(OPS))]; s = np.r_[neg, np.concatenate([pos[o] for o in OPS])]
    f, t, _ = roc_curve(y, s)
    out[m] = dict(auc=round(float(pt), 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                  per_edit=per, tpr10=round(float(np.interp(0.10, f, t)), 3), ctrl_flag=round(float((neg > 0.5).mean()), 3) if m.startswith("gbm") else round(float((neg > 0).mean()), 3),
                  forged_flag=round(float(np.mean(np.concatenate([pos[o] for o in OPS]) > (0.5 if m.startswith("gbm") else 0))), 3))
    print(m, out[m], flush=True)
json.dump(out, open(sys.argv[3], "w"), indent=1)
