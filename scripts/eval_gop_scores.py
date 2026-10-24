"""Per-device scores of the paper's detectors on the GOP-matched tiers.
  FEATSET=v2 PAIR=1 LOCAL=0 python scripts/eval_gop_scores.py base <out.pkl>   (snapshot_conflict, drift_only)
  FEATSET=v2 PAIR=1 LOCAL=1 python scripts/eval_gop_scores.py gbm  <out.pkl>   (supervised classifier with local features)
The classifier is trained exactly as in the paper (naive + platform tiers, leave-one-device-out) and never sees the GOP-matched files."""
import os, sys, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, "scripts")
import eval_v2 as E

mode, out = sys.argv[1], sys.argv[2]
NEW = ("platform_gop", "simctl_gop")
gm = pickle.load(open("work/gop_matched_feats.pkl", "rb"))
for p, d in gm.items():
    key = p.replace(chr(92), "/"); nat = d["native"].replace(chr(92), "/")
    E._EXT[key] = d["ext"]; E._PAIRS[(nat, key)] = d["pair"]; E._LOC[(nat, key)] = d["loc"]

def newfiles():
    out_ = {}
    for i, t in enumerate(E.T):
        nat = t["native"]
        p = E.forged_path("simctl_gop", i, None)
        if p in E._EXT: out_[("simctl_gop", i, None)] = E.edge_vec(nat, p)
        for op in E.OPS:
            p = E.forged_path("platform_gop", i, op)
            if p in E._EXT: out_[("platform_gop", i, op)] = E.edge_vec(nat, p)
    return out_

newf = newfiles()
print({t: sum(1 for k in newf if k[0] == t) for t in NEW})
res = {}
if mode == "base":
    dets = E.fold_detectors()
    for m in ("snapshot_conflict", "drift_only"):
        res[m] = {"genuine": {}, **{t: {} for t in NEW}}
    for d in E.DEVS:
        te = np.flatnonzero(E.DEV == d)
        s = dets[d].scores(E.N[te], E.W[te])
        for m in res: res[m]["genuine"][d] = s[m]
        for t in NEW:
            ks = [k for k in newf if k[0] == t and E.DEV[k[1]] == d]
            if ks:
                s = dets[d].scores(E.N[[k[1] for k in ks]], np.array([newf[k] for k in ks]))
                for m in res: res[m][t][d] = s[m]
            else:
                for m in res: res[m][t][d] = np.array([])
else:
    TRAIN = ("naive", "platform")
    forged = E.load_forged(TRAIN)
    Xg, dg = E.W, E.DEV
    def stack(tiers):
        ks = [k for k in forged if k[0] in tiers]
        return np.array([forged[k] for k in ks]), np.array([E.DEV[k[1]] for k in ks])
    g, p = {}, {t: {} for t in NEW}
    for d in E.DEVS:
        te = dg == d
        Xp, dp = stack(TRAIN)
        X = np.vstack([Xg[~te], Xp[dp != d]]); y = np.r_[np.zeros((~te).sum()), np.ones((dp != d).sum())]
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, class_weight="balanced", random_state=0).fit(X, y)
        g[d] = clf.predict_proba(Xg[te])[:, 1]
        for t in NEW:
            ks = [k for k in newf if k[0] == t and E.DEV[k[1]] == d]
            p[t][d] = clf.predict_proba(np.array([newf[k] for k in ks]))[:, 1] if ks else np.array([])
    res["gbm"] = dict(genuine=g, **p)
pickle.dump(res, open(out, "wb")); print("saved", out)
