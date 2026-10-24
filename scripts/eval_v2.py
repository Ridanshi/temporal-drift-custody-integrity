"""Single-hop evaluation on real forged files, leave-one-device-out."""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from custody.detectors_v2 import Detector, METHODS
from custody.forge import OPS
from custody.vision_loader import discover_triplets

import os
ROOT = "E:/DF paper/data/vision"
FEATSET = os.environ.get("FEATSET", "ext")
NCOL = 12 if FEATSET == "v2" else 18
T = discover_triplets(ROOT)
import sys as _s; _s.path.insert(0, 'scripts'); import srcshim; srcshim.apply(T)
_EXT = {k.replace(chr(92), "/"): v for k, v in np.load("work/features_ext.npz", allow_pickle=True)["features"].item().items()}


PAIR = os.environ.get("PAIR", "1") == "1"
_PAIRS = {}
if PAIR:
    _PAIRS = {(k[0].replace(chr(92), "/"), k[1].replace(chr(92), "/")): v
              for k, v in np.load(os.environ.get("PAIRC","work/pair_features.npz"), allow_pickle=True)["features"].item().items()}
LOCAL = os.environ.get("LOCAL", "0") == "1"
_LOC = {}
if LOCAL:
    _LOC = {(k[0].replace(chr(92), "/"), k[1].replace(chr(92), "/")): v for k, v in np.load(os.environ.get("LOCC","work/local_features.npz"), allow_pickle=True)["features"].item().items()}
NP = (5 if PAIR else 0) + (8 if LOCAL else 0)


def feat(path):
    return _EXT[path.replace(chr(92), "/")][:NCOL]


def edge_vec(src, tgt):
    """Target vector = file features (+ pair features if enabled); native side padded with zeros."""
    v = feat(tgt)
    key = (src.replace(chr(92), "/"), tgt.replace(chr(92), "/"))
    if PAIR:
        v = np.r_[v, _PAIRS[key]]
    if LOCAL:
        v = np.r_[v, _LOC.get(key, np.zeros(8))]
    return v


def nat_vec(src):
    v = feat(src)
    return np.r_[v, np.zeros(NP)] if PAIR else v


N = np.array([nat_vec(t["native"]) for t in T]); W = np.array([edge_vec(t["native"], t["whatsapp"]) for t in T]); Y = np.array([edge_vec(t["native"], t["youtube"]) for t in T])
DEV = np.array([t["device"] for t in T])
DEVS = sorted(set(DEV))
EXTRA = ("text", "inpaint", "cut")


def forged_path(tier, i, op):
    return f"work/forged/{tier}/{i:03d}_{op or 'none'}.mp4"


def load_forged(tiers=("simctl", "platform", "naive", "adaptive", "adaptive_drift_only", "adaptive_snapshot_conflict", "s25", "s50", "platform_x", "adaptive_fusion", "adaptive_gbm")):
    out = {}
    for tier in tiers:
        for i in range(len(T)):
            for op in ([None] if tier == "simctl" else (("text", "inpaint", "cut") if tier == "platform_x" else OPS)):
                p = forged_path(tier, i, op)
                if p in _EXT and (not PAIR or (T[i]["native"].replace(chr(92), "/"), p) in _PAIRS) and (not LOCAL or (T[i]["native"].replace(chr(92), "/"), p) in _LOC):
                    out[(tier, i, op)] = edge_vec(T[i]["native"], p)
    return out


def fold_detectors():
    return {d: Detector(joint_cols=list(range(NCOL))).fit(N[DEV != d], W[DEV != d], DEV[DEV != d]) for d in DEVS}


def collect(forged, tiers, dets=None):
    """-> dict[group] -> dict[method] -> dict[dev] -> scores ; groups: 'genuine','simctl',(tier,op)."""
    dets = dets or fold_detectors()
    out = {}
    def put(g, d, s):
        for m in METHODS:
            out.setdefault(g, {}).setdefault(m, {})[d] = s[m]
    for d in DEVS:
        te = np.flatnonzero(DEV == d)
        put("genuine", d, dets[d].scores(N[te], W[te]))
        ks = [("simctl", i, None) for i in te if ("simctl", i, None) in forged]
        if ks:
            put("simctl", d, dets[d].scores(N[[k[1] for k in ks]], np.array([forged[k] for k in ks])))
        for tier in tiers:
            for op in (EXTRA if tier == "platform_x" else OPS):
                ks = [(tier, i, op) for i in te if (tier, i, op) in forged]
                if ks:
                    put((tier, op), d, dets[d].scores(N[[k[1] for k in ks]], np.array([forged[k] for k in ks])))
    return out


def pooled(sc, g, m, devs=None):
    devs = devs or list(sc[g][m].keys())
    return np.concatenate([sc[g][m][d] for d in devs if d in sc[g][m]])


def auc_ci(sc, pos_groups, m, n_boot=1000, seed=0):
    """Pooled AUC of genuine vs union of pos_groups, with device-cluster bootstrap CI."""
    rng = np.random.default_rng(seed)
    def auc(devs):
        g = pooled(sc, "genuine", m, devs)
        p = np.concatenate([pooled(sc, pg, m, devs) for pg in pos_groups if pg in sc and m in sc[pg]])
        if len(g) == 0 or len(p) == 0:
            return np.nan
        return roc_auc_score(np.r_[np.zeros(len(g)), np.ones(len(p))], np.r_[g, p])
    point = auc(DEVS)
    bs = [auc(list(rng.choice(DEVS, len(DEVS), replace=True))) for _ in range(n_boot)]
    bs = np.array([b for b in bs if not np.isnan(b)])
    return float(point), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def tpr_at_fpr(sc, pos_groups, m, fpr=0.10):
    g = pooled(sc, "genuine", m)
    p = np.concatenate([pooled(sc, pg, m) for pg in pos_groups if pg in sc])
    f, t, _ = roc_curve(np.r_[np.zeros(len(g)), np.ones(len(p))], np.r_[g, p])
    return float(np.interp(fpr, f, t))


def flag_rates(sc, pos_groups, m):
    g = pooled(sc, "genuine", m); p = np.concatenate([pooled(sc, pg, m) for pg in pos_groups if pg in sc])
    return float((g > 0).mean()), float((p > 0).mean())


if __name__ == "__main__":
    forged = load_forged()
    tiers = [t for t in ("naive", "platform", "adaptive", "adaptive_drift_only", "adaptive_snapshot_conflict", "s25", "s50") if any(k[0] == t for k in forged)]
    sc = collect(forged, tiers)
    res = {}
    print("%-9s %-18s %-22s %6s %6s %6s" % ("tier", "method", "AUC [95% CI]", "TPR@10", "FPR", "TPR"))
    for tier in tiers + ["simctl"]:
        pg = [("simctl")] if tier == "simctl" else [(tier, op) for op in OPS]
        for m in METHODS:
            a, lo, hi = auc_ci(sc, pg, m)
            fpr, tpr = flag_rates(sc, pg, m)
            res.setdefault(tier, {})[m] = dict(auc=a, lo=lo, hi=hi, tpr_at_10=tpr_at_fpr(sc, pg, m), fpr=fpr, tpr=tpr)
            print("%-9s %-18s %.3f [%.3f,%.3f]    %6.3f %6.3f %6.3f" % (tier, m, a, lo, hi, res[tier][m]["tpr_at_10"], fpr, tpr))
    json.dump(res, open(f"scripts/eval_v2_singlehop_{FEATSET}{'_pair' if PAIR else ''}{'_local' if LOCAL else ''}.json", "w"), indent=1)
