"""Supervised gradient-boosting detector, tested leave-device-out and leave-one-edit-out."""
import json, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, "scripts")
import eval_v2 as E

TRAIN_TIERS = ("naive", "platform")
import os
TEST_TIERS = ["naive", "platform", "adaptive", "adaptive_drift_only", "adaptive_fusion"] + (["adaptive_gbm"] if os.path.exists("work/forged/adaptive_gbm") and os.environ.get("USE_GBM_TIER") else [])
forged = E.load_forged(TRAIN_TIERS + tuple(TEST_TIERS[2:]))
Xg, dg = E.W, E.DEV

def stack(tiers, ops=None):
    ks = [k for k in forged if k[0] in tiers and (ops is None or k[2] in ops)]
    return np.array([forged[k] for k in ks]), np.array([E.DEV[k[1]] for k in ks]), ks

def fit(Xp, Xn):
    X = np.vstack([Xn, Xp]); y = np.r_[np.zeros(len(Xn)), np.ones(len(Xp))]
    return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, class_weight="balanced", random_state=0).fit(X, y)

def run(leave_edit=None):
    sc_g, sc_p = [], {t: [] for t in TEST_TIERS}
    for d in E.DEVS:
        te = dg == d
        train_ops = [o for o in E.OPS if o != leave_edit]
        Xp, dp, _ = stack(TRAIN_TIERS, train_ops)
        clf = fit(Xp[dp != d], Xg[~te])
        for t in TEST_TIERS:
            ops = [leave_edit] if leave_edit else None
            Xt, dt, ks = stack((t,), ops)
            m = dt == d
            if m.any(): sc_p[t].append(clf.predict_proba(Xt[m])[:, 1])
        if leave_edit is None or True: sc_g.append((d, clf.predict_proba(Xg[te])[:, 1]))
    return sc_g, sc_p

out = {}
rng = np.random.default_rng(0)
for mode in (None,) + tuple(E.OPS):
    sc_g, sc_p = run(mode)
    gd = {d: x for d, x in sc_g}
    g = np.concatenate([x for _, x in sc_g])
    r = {}
    for t in TEST_TIERS:
        if not sc_p[t]: continue
        ks_t = stack((t,), [mode] if mode else None)
        devs_t = ks_t[1]
        # per-device positive scores (re-split by device in the order they were appended)
        order = [d for d in E.DEVS if (devs_t == d).any()]
        pd_ = dict(zip(order, sc_p[t]))
        p = np.concatenate(sc_p[t])
        a = roc_auc_score(np.r_[np.zeros(len(g)), np.ones(len(p))], np.r_[g, p])
        f, tp, _ = roc_curve(np.r_[np.zeros(len(g)), np.ones(len(p))], np.r_[g, p])
        bs = []
        if mode is None:
            for _ in range(500):
                ds = list(rng.choice(E.DEVS, len(E.DEVS), replace=True))
                gg = np.concatenate([gd[x] for x in ds]); pp = [pd_[x] for x in ds if x in pd_]
                if pp: bs.append(roc_auc_score(np.r_[np.zeros(len(gg)), np.ones(sum(len(q) for q in pp))], np.r_[gg, np.concatenate(pp)]))
        r[t] = dict(auc=float(a), lo=float(np.percentile(bs, 2.5)) if bs else None, hi=float(np.percentile(bs, 97.5)) if bs else None,
                    tpr10=float(np.interp(0.10, f, tp)), tpr1=float(np.interp(0.01, f, tp)))
    out["all_edits" if mode is None else f"leave_out_{mode}"] = r
    print(mode or "all_edits", {t: round(v["auc"], 3) for t, v in r.items()}, flush=True)
json.dump(out, open("scripts/classifier_v2_%s.json" % ("local" if E.LOCAL else "base"), "w"), indent=1)
