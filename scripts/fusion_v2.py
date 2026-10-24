"""Learned fusion of component scores, trained on other devices only (nested leave-device-out)."""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "scripts")
import eval_v2 as E
from custody.detectors_v2 import Detector

forged = E.load_forged(); forged.update(E.load_forged(("adaptive_fusion",)))
FE = ["drift_only", "coevasion_only", "mismatch_only", "snapshot_conflict", "isolation_forest", "ocsvm"]
VARIANTS = {"fusion_all_np": (FE, ("naive", "platform")), "fusion_dc_np": (["drift_only", "coevasion_only"], ("naive", "platform")),
            "fusion_all_n": (FE, ("naive",))}
import os
TEST = ["naive", "platform", "adaptive", "adaptive_drift_only"] + (["adaptive_fusion"] if os.path.exists("work/forged/adaptive_fusion") else [])

def groups_for(dev_mask_idx, tiers, with_ctl=False):
    out = {"genuine": (E.N[dev_mask_idx], E.W[dev_mask_idx])}
    for t in tiers:
        ks = [(t, i, op) for i in dev_mask_idx for op in E.OPS if (t, i, op) in forged]
        if ks:
            out[t] = (E.N[[k[1] for k in ks]], np.array([forged[k] for k in ks]))
    return out

def mat(det, nat, tgt, cols):
    s = det.scores(nat, tgt)
    return np.column_stack([s[c] for c in cols])

sc = {}
COEF = {}
for d in E.DEVS:
    te = np.flatnonzero(E.DEV == d)
    det_d = Detector(joint_cols=list(range(E.NCOL))).fit(E.N[E.DEV != d], E.W[E.DEV != d], E.DEV[E.DEV != d])
    train = {}
    for e in E.DEVS:
        if e == d: continue
        m = (E.DEV != d) & (E.DEV != e)
        det_de = Detector(joint_cols=list(range(E.NCOL))).fit(E.N[m], E.W[m], E.DEV[m])
        g = groups_for(np.flatnonzero(E.DEV == e), ("naive", "platform"))
        for k, (n_, t_) in g.items():
            train.setdefault(k, []).append(mat(det_de, n_, t_, FE))
    train = {k: np.vstack(v) for k, v in train.items()}
    tg = groups_for(te, TEST)
    for name, (cols, ttiers) in VARIANTS.items():
        ci = [FE.index(c) for c in cols]
        X = np.vstack([train["genuine"][:, ci]] + [train[t][:, ci] for t in ttiers])
        y = np.r_[np.zeros(len(train["genuine"])), np.ones(sum(len(train[t]) for t in ttiers))]
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X, y)
        if name == "fusion_all_np": COEF[d] = dict(coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), cols=cols)
        for k, (n_, t_) in tg.items():
            g = "genuine" if k == "genuine" else (k, None)
            sc_ = lr.decision_function(mat(det_d, n_, t_, FE)[:, ci])
            sc.setdefault(k, {}).setdefault(name, {})[d] = sc_
    print("done", d, flush=True)

from sklearn.metrics import roc_auc_score
out = {}
for name in VARIANTS:
    out[name] = {}
    for t in TEST:
        g = np.concatenate([sc["genuine"][name][d] for d in E.DEVS]); p = np.concatenate([sc[t][name][d] for d in E.DEVS if d in sc[t][name]])
        auc = roc_auc_score(np.r_[np.zeros(len(g)), np.ones(len(p))], np.r_[g, p])
        rng = np.random.default_rng(0); bs = []
        for _ in range(1000):
            ds = list(rng.choice(E.DEVS, len(E.DEVS), replace=True))
            gg = np.concatenate([sc["genuine"][name][x] for x in ds]); pp = np.concatenate([sc[t][name][x] for x in ds if x in sc[t][name]])
            if len(pp): bs.append(roc_auc_score(np.r_[np.zeros(len(gg)), np.ones(len(pp))], np.r_[gg, pp]))
        from sklearn.metrics import roc_curve
        f, tp, _ = roc_curve(np.r_[np.zeros(len(g)), np.ones(len(p))], np.r_[g, p])
        out[name][t] = dict(auc=float(auc), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), tpr10=float(np.interp(0.10, f, tp)))
        print("%-14s %-20s AUC %.3f [%.3f,%.3f] TPR@10 %.3f" % (name, t, auc, out[name][t]["lo"], out[name][t]["hi"], out[name][t]["tpr10"]))
json.dump(out, open("scripts/fusion_v2.json", "w"), indent=1)
json.dump(COEF, open("work/fusion_coefs.json", "w"))
