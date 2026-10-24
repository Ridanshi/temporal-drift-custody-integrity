"""Per-device scores for the two revision experiments (read-only on existing data).

  gbm_ctrl : PAIR=1 LOCAL=1  classifier trained with untampered control encodes (simctl) as extra genuine class
  nosrc    : PAIR=0 LOCAL=0  detectors that never see the source file (single-file features only)

  python scripts/eval_fixes.py <gbm_ctrl|nosrc> <out.pkl>
"""
import os, sys, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, "scripts")
import eval_v2 as E

mode, out = sys.argv[1], sys.argv[2]
TEST = ("naive", "platform", "adaptive", "adaptive_drift_only", "adaptive_gbm", "simctl", "platform_gop", "simctl_gop")
gm = pickle.load(open("work/gop_matched_feats.pkl", "rb"))
for p, d in gm.items():
    key = p.replace(chr(92), "/"); nat = d["native"].replace(chr(92), "/")
    E._EXT[key] = d["ext"]; E._PAIRS[(nat, key)] = d["pair"]; E._LOC[(nat, key)] = d["loc"]
forged = E.load_forged(("naive", "platform", "adaptive", "adaptive_drift_only", "adaptive_gbm", "simctl"))
for i, t in enumerate(E.T):
    p = E.forged_path("simctl_gop", i, None)
    if p in E._EXT: forged[("simctl_gop", i, None)] = E.edge_vec(t["native"], p)
    for op in E.OPS:
        p = E.forged_path("platform_gop", i, op)
        if p in E._EXT: forged[("platform_gop", i, op)] = E.edge_vec(t["native"], p)
print({t: sum(1 for k in forged if k[0] == t) for t in TEST}, "NP", E.NP)

def stack(tiers, excl_dev=None, only_dev=None):
    ks = [k for k in forged if k[0] in tiers and (excl_dev is None or E.DEV[k[1]] != excl_dev) and (only_dev is None or E.DEV[k[1]] == only_dev)]
    return (np.array([forged[k] for k in ks]) if ks else np.zeros((0, E.W.shape[1]))), ks

def gbm(X, y):
    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, class_weight="balanced", random_state=0).fit(X, y)

res = {}
def put(m, grp, d, s):
    res.setdefault(m, {}).setdefault(grp, {})[d] = s

dets = E.fold_detectors() if mode == "nosrc" else None
for d in E.DEVS:
    te = E.DEV == d
    Xg_tr = E.W[~te]
    Xp, _ = stack(("naive", "platform"), excl_dev=d)
    Xc, _ = stack(("simctl",), excl_dev=d)
    variants = {"gbm_orig": (np.vstack([Xg_tr]), Xp), "gbm_ctrl": (np.vstack([Xg_tr, Xc]), Xp)}
    clfs = {m: gbm(np.vstack([neg, pos]), np.r_[np.zeros(len(neg)), np.ones(len(pos))]) for m, (neg, pos) in variants.items()}
    for m, clf in clfs.items():
        put(m, "genuine", d, clf.predict_proba(E.W[te])[:, 1])
        for t in TEST:
            X, _ = stack((t,), only_dev=d)
            put(m, t, d, clf.predict_proba(X)[:, 1] if len(X) else np.array([]))
    if dets:
        idx = np.flatnonzero(te); s = dets[d].scores(E.N[idx], E.W[idx])
        for m in ("snapshot_conflict", "drift_only"): put(m, "genuine", d, s[m])
        for t in TEST:
            X, ks = stack((t,), only_dev=d)
            if len(X):
                s = dets[d].scores(E.N[[k[1] for k in ks]], X)
                for m in ("snapshot_conflict", "drift_only"): put(m, t, d, s[m])
            else:
                for m in ("snapshot_conflict", "drift_only"): put(m, t, d, np.array([]))
pickle.dump(res, open(out, "wb")); print("saved", out, list(res))
