"""Score REAL WhatsApp copies of the pilot clips with models trained on VISION only.
Input: work/real_pilot_map.csv (clip,condition,source,received) -> prints per-file scores and the VISION genuine reference range.
  FEATSET=v2 PAIR=1 LOCAL=1 python scripts/real_pilot_score.py <map.csv> <out.pkl>"""
import os, sys, csv, pickle
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, "scripts")
import eval_v2 as E
from custody.features_v2 import extract_features_ext, pair_features_multi
from custody.local_features import decode_small, local_pair_features
from custody.detectors_v2 import Detector

rows = list(csv.DictReader(open(sys.argv[1])))
by_src = {}
for r in rows: by_src.setdefault(r["source"], []).append(r)
rec = {}
for src, rs in by_src.items():
    files = [r["received"] for r in rs]
    pairs = pair_features_multi(src, files); sm = decode_small(src)
    for r, pv in zip(rs, pairs):
        rec[(r["clip"], r["condition"])] = dict(ext=extract_features_ext(r["received"]), pair=pv, loc=local_pair_features(sm, decode_small(r["received"])), src_ext=extract_features_ext(src))
NC = E.NCOL
def vec(d, loc): 
    v = np.r_[d["ext"][:NC], d["pair"]]
    return np.r_[v, d["loc"]] if loc else v
Nv = lambda d: np.r_[d["src_ext"][:NC], np.zeros(5)]
keys = list(rec)
det = Detector(joint_cols=list(range(NC))).fit(E.N, E.W, E.DEV)
res = {}
if not E.LOCAL:
    s = det.scores(np.array([Nv(rec[k]) for k in keys]), np.array([vec(rec[k], False) for k in keys]))
    for m in ("snapshot_conflict", "drift_only"): res[m] = dict(zip(keys, np.asarray(s[m])))
    g = det.scores(E.N, E.W)
    res["vision_genuine_ref"] = {m: (float(np.percentile(g[m], 50)), float(np.percentile(g[m], 90)), float(np.percentile(g[m], 99))) for m in ("snapshot_conflict", "drift_only")}
else:
    forged = E.load_forged(("naive", "platform", "simctl"))
    Xp = np.array([forged[k] for k in forged if k[0] in ("naive", "platform")]); Xc = np.array([forged[k] for k in forged if k[0] == "simctl"])
    for name, neg in (("gbm_orig", E.W), ("gbm_ctrl", np.vstack([E.W, Xc]))):
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, class_weight="balanced", random_state=0).fit(
            np.vstack([neg, Xp]), np.r_[np.zeros(len(neg)), np.ones(len(Xp))])
        res[name] = dict(zip(keys, clf.predict_proba(np.array([vec(rec[k], True) for k in keys]))[:, 1]))
        res[name + "_vision_genuine_p90"] = float(np.percentile(clf.predict_proba(E.W)[:, 1], 90))
pickle.dump(res, open(sys.argv[2], "wb"))
for m, v in res.items():
    if isinstance(v, dict): print(m, {f"{c}/{o}": round(float(x), 3) for (c, o), x in v.items()})
    else: print(m, v)
