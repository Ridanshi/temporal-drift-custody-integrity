"""Native source vs degraded (YouTube-copy) source, same detectors, leave-one-device-out.
  FEATSET=v2 PAIR=1 LOCAL=0 python scripts/degraded_eval.py base out.json            (SRC unset = native; SRC=youtube + PAIRC/LOCC = degraded)
  FEATSET=v2 PAIR=1 LOCAL=1 python scripts/degraded_eval.py gbm  out.json
Positives: platform-aware (5 edits). Negatives: real WhatsApp copy ('genuine') and untampered control ('simctl' = edit-only)."""
import sys, json
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, "scripts")
import eval_v2 as E

mode, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
forged = E.load_forged(("naive", "platform", "simctl"))
res = {}

def summarize(sc, name):
    """sc[group][dev] -> scores; groups genuine, simctl, platform, naive"""
    r = {}
    for neg in ("genuine", "simctl"):
        for pos in ("platform", "naive"):
            def auc(devs):
                n = np.concatenate([sc[neg][d] for d in devs if d in sc[neg]]); p = np.concatenate([sc[pos][d] for d in devs if d in sc[pos]])
                return roc_auc_score(np.r_[np.zeros(len(n)), np.ones(len(p))], np.r_[n, p])
            pt = auc(E.DEVS); bs = [auc(list(rng.choice(E.DEVS, len(E.DEVS), replace=True))) for _ in range(1000)]
            r[f"{neg}|{pos}"] = [round(pt, 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    res[name] = r; print(name, r, flush=True)

if mode == "base":
    dets = E.fold_detectors()
    sc = {m: {"genuine": {}, "simctl": {}, "platform": {}, "naive": {}} for m in ("snapshot_conflict", "drift_only")}
    for d in E.DEVS:
        te = np.flatnonzero(E.DEV == d)
        s = dets[d].scores(E.N[te], E.W[te])
        for m in sc: sc[m]["genuine"][d] = np.asarray(s[m])
        for grp, tiers in (("simctl", ("simctl",)), ("platform", ("platform",)), ("naive", ("naive",))):
            ks = [k for k in forged if k[0] in tiers and E.DEV[k[1]] == d]
            if not ks: continue
            s = dets[d].scores(E.N[[k[1] for k in ks]], np.array([forged[k] for k in ks]))
            for m in sc: sc[m][grp][d] = np.asarray(s[m])
    for m in sc: summarize(sc[m], m)
else:
    def stack(tiers, excl=None, only=None):
        ks = [k for k in forged if k[0] in tiers and (excl is None or E.DEV[k[1]] != excl) and (only is None or E.DEV[k[1]] == only)]
        return (np.array([forged[k] for k in ks]) if ks else np.zeros((0, E.W.shape[1])))
    def gbm(neg, pos):
        return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, class_weight="balanced", random_state=0).fit(
            np.vstack([neg, pos]), np.r_[np.zeros(len(neg)), np.ones(len(pos))])
    sc = {m: {"genuine": {}, "simctl": {}, "platform": {}, "naive": {}} for m in ("gbm_orig", "gbm_ctrl")}
    for d in E.DEVS:
        te = E.DEV == d
        Xp = stack(("naive", "platform"), excl=d); Xc = stack(("simctl",), excl=d)
        for m, neg in (("gbm_orig", E.W[~te]), ("gbm_ctrl", np.vstack([E.W[~te], Xc]))):
            clf = gbm(neg, Xp)
            sc[m]["genuine"][d] = clf.predict_proba(E.W[te])[:, 1]
            for grp in ("simctl", "platform", "naive"):
                X = stack((grp,), only=d)
                if len(X): sc[m][grp][d] = clf.predict_proba(X)[:, 1]
    for m in sc: summarize(sc[m], m)
json.dump(res, open(out, "w"), indent=1)
