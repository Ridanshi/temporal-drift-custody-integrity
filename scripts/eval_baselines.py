"""Evaluate the established-method baselines (scripts/baselines_established.py) leave-one-device-out and compare them with the
paper's detectors by paired device-cluster bootstrap.

  python scripts/eval_baselines.py <p_base.pkl> <p_gbm.pkl> <out.json>

p_base.pkl / p_gbm.pkl hold per-device scores of snapshot_conflict, drift_only and the supervised classifier (same genuine/forged files).
"""
import os, sys, glob, pickle, json
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve

os.environ.setdefault("FEATSET", "v2"); os.environ.setdefault("PAIR", "1"); os.environ.setdefault("LOCAL", "0")
sys.path.insert(0, "scripts")
import eval_v2 as E

pb, pg, outp = sys.argv[1:4]
extra = sys.argv[4:6]
F = {}
for f in sorted(glob.glob("work/baselines_v2/clip_*.pkl")) + sorted(glob.glob("work/baselines_gop/clip_*.pkl")):
    F.update(pickle.load(open(f, "rb")))
GROUPS = ("gop", "dq", "prnu", "temp")
TIERS = ["naive", "platform", "adaptive", "adaptive_drift_only", "adaptive_gbm", "simctl", "platform_gop", "simctl_gop"]
DEV = E.DEV; DEVS = E.DEVS; n = len(E.T)
assert all(("genuine", i, None) in F for i in range(n)), "extraction incomplete"
keys = {t: [k for k in F if k[0] == t] for t in TIERS}
print({t: len(v) for t, v in keys.items()})

def vec(k, g):
    return F[k][g]

def fit_score(train_X):
    mu = train_X.mean(0); sd = train_X.std(0) + 1e-9
    Z = (train_X - mu) / sd
    lw = LedoitWolf().fit(Z); P = lw.precision_
    def d(X):
        Y = (X - mu) / sd
        return np.sqrt(np.einsum("ij,jk,ik->i", Y, P, Y))
    dtr = d(train_X); tau = np.percentile(dtr, 90); sc = max(np.percentile(dtr, 75) - np.percentile(dtr, 25), 1e-6) / 1.349
    return lambda X: (d(X) - tau) / sc

methods = list(GROUPS) + ["established_max", "established_joint"]
G = {m: {} for m in methods}; P = {m: {t: {} for t in TIERS} for m in methods}
for dv in DEVS:
    tr = [i for i in range(n) if DEV[i] != dv]; te = [i for i in range(n) if DEV[i] == dv]
    scorers = {g: fit_score(np.array([vec(("genuine", i, None), g) for i in tr])) for g in GROUPS}
    cat = lambda k: np.concatenate([vec(k, g) for g in GROUPS])
    joint = fit_score(np.array([cat(("genuine", i, None)) for i in tr]))
    def score(klist):
        out = {}
        for g in GROUPS:
            out[g] = scorers[g](np.array([vec(k, g) for k in klist])) if klist else np.array([])
        out["established_joint"] = joint(np.array([cat(k) for k in klist])) if klist else np.array([])
        out["established_max"] = np.max([out[g] for g in GROUPS], axis=0) if klist else np.array([])
        return out
    s = score([("genuine", i, None) for i in te])
    for m in methods: G[m][dv] = s[m]
    for t in TIERS:
        kl = [k for k in keys[t] if DEV[k[1]] == dv]
        s = score(kl)
        for m in methods: P[m][t][dv] = s[m]

def auc_of(g, p, devs):
    a = np.concatenate([g[d] for d in devs]); b = np.concatenate([p[d] for d in devs])
    return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])

def tpr10(g, p):
    a = np.concatenate([g[d] for d in DEVS]); b = np.concatenate([p[d] for d in DEVS])
    f, t, _ = roc_curve(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]); return float(np.interp(0.10, f, t))

rng = np.random.default_rng(0)
boots = [list(rng.choice(DEVS, len(DEVS), replace=True)) for _ in range(2000)]
base = pickle.load(open(pb, "rb")); gbm = pickle.load(open(pg, "rb"))["gbm"]
if extra:
    b4 = pickle.load(open(extra[0], "rb")); g4 = pickle.load(open(extra[1], "rb"))["gbm"]
    for m in b4: base[m].update({k: v for k, v in b4[m].items() if k != "genuine"})
    gbm.update({k: v for k, v in g4.items() if k != "genuine"})
others = {"snapshot_conflict": (base["snapshot_conflict"]["genuine"], {t: base["snapshot_conflict"][t] for t in TIERS if t in base["snapshot_conflict"]}),
          "drift_only": (base["drift_only"]["genuine"], {t: base["drift_only"][t] for t in TIERS if t in base["drift_only"]}),
          "gbm_local": (gbm["genuine"], {t: gbm[t] for t in TIERS if t in gbm})}
res = {}
for t in TIERS:
    if not keys[t]: continue
    for m in methods:
        g, p = G[m], P[m][t]
        a = auc_of(g, p, DEVS); bs = np.array([auc_of(g, p, b) for b in boots])
        res[f"{t}|{m}"] = dict(auc=round(a, 3), lo=round(float(np.percentile(bs, 2.5)), 3), hi=round(float(np.percentile(bs, 97.5)), 3), tpr10=round(tpr10(g, p), 3))
        for on, (og, op) in others.items():
            if t in op:
                diff = np.array([auc_of(og, op[t], b) - auc_of(g, p, b) for b in boots]); d0 = auc_of(og, op[t], DEVS) - a
                res[f"{t}|{on}_minus_{m}"] = dict(diff=round(d0, 3), lo=round(float(np.percentile(diff, 2.5)), 3), hi=round(float(np.percentile(diff, 97.5)), 3))
json.dump(res, open(outp, "w"), indent=1)
for t in TIERS:
    print("\n==", t)
    for m in methods:
        k = f"{t}|{m}"
        if k in res: r = res[k]; print(f"  {m:18s} AUC {r['auc']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]  TPR@10% {r['tpr10']:.2f}")


# ---- pipeline-controlled comparison: negatives are the untampered control encodes (same pipeline, no edit), positives the forged files.
# This isolates edit detection from any difference between our encoder pipeline and the real platform's.
ctrl = {}
allm = {m: (G[m], P[m]) for m in methods}
allm.update({"snapshot_conflict": (base["snapshot_conflict"]["genuine"], base["snapshot_conflict"]),
             "drift_only": (base["drift_only"]["genuine"], base["drift_only"]),
             "gbm_local": (gbm["genuine"], gbm)})
for t in ("naive", "platform", "adaptive", "adaptive_drift_only", "adaptive_gbm", "platform_gop"):
    for m, (g, pp) in allm.items():
        neg, pos = pp["simctl_gop" if t == "platform_gop" else "simctl"], pp[t]
        a = auc_of(neg, pos, DEVS); bs = np.array([auc_of(neg, pos, b) for b in boots])
        ctrl[f"{t}|{m}"] = dict(auc=round(a, 3), lo=round(float(np.percentile(bs, 2.5)), 3), hi=round(float(np.percentile(bs, 97.5)), 3))
json.dump(ctrl, open(outp.replace(".json", "_controlled.json"), "w"), indent=1)
print("\n== pipeline-controlled (forged vs untampered control encodes)")
for t in ("naive", "platform", "adaptive", "adaptive_gbm", "platform_gop"):
    print(" ", t, " | ".join(f"{m} {ctrl[f'{t}|{m}']['auc']:.2f}" for m in allm))
