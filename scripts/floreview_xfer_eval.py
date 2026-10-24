"""Train on VISION (all 15 devices), score the FloreView transfer set (43 unseen devices).
  FEATSET=v2 PAIR=1 LOCAL=0 python scripts/floreview_xfer_eval.py base <out.pkl>   unsupervised detectors
  FEATSET=v2 PAIR=1 LOCAL=1 python scripts/floreview_xfer_eval.py gbm  <out.pkl>   supervised classifier (original and control-trained)
Negatives = untampered pipeline control of the same clip; positives = the five edits. Output: scores[model][group] = array (clip order)."""
import os, sys, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, "scripts")
import eval_v2 as E
from custody.forge import OPS
from custody.detectors_v2 import Detector

mode, out = sys.argv[1], sys.argv[2]
fv = pickle.load(open("work/floreview_feats.pkl", "rb"))
idx = sorted(fv)
NC = E.NCOL

def vec(rec):
    v = rec["ext"][:NC]
    if E.PAIR: v = np.r_[v, rec["pair"]]
    if E.LOCAL: v = np.r_[v, rec["loc"]]
    return v

Nfv = np.array([np.r_[fv[i]["native_ext"][:NC], np.zeros(E.NP)] if E.PAIR else fv[i]["native_ext"][:NC] for i in idx])
groups = {"ctrl": np.array([vec(fv[i]["ctrl"]) for i in idx])}
for op in OPS: groups[op] = np.array([vec(fv[i][op]) for i in idx])
dev = np.array([fv[i]["device"] for i in idx])
res = {"device": dev}

if mode == "base":
    det = Detector(joint_cols=list(range(NC))).fit(E.N, E.W, E.DEV)
    for g, X in groups.items():
        s = det.scores(Nfv, X)
        for m in ("snapshot_conflict", "drift_only"): res.setdefault(m, {})[g] = np.asarray(s[m])
else:
    forged = E.load_forged(("naive", "platform", "simctl"))
    def stack(tiers):
        ks = [k for k in forged if k[0] in tiers]
        return np.array([forged[k] for k in ks])
    Xp, Xc = stack(("naive", "platform")), stack(("simctl",))
    def gbm(neg, pos):
        X = np.vstack([neg, pos]); y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, class_weight="balanced", random_state=0).fit(X, y)
    for name, neg in (("gbm_orig", E.W), ("gbm_ctrl", np.vstack([E.W, Xc]))):
        clf = gbm(neg, Xp)
        for g, X in groups.items(): res.setdefault(name, {})[g] = clf.predict_proba(X)[:, 1]
pickle.dump(res, open(out, "wb")); print("saved", out, [k for k in res if k != "device"], len(idx), "clips")
