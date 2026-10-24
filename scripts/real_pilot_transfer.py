"""VISION-trained models applied unchanged to ALL usable real-WhatsApp pilot files: how often is each condition flagged?
  FEATSET=v2 PAIR=1 LOCAL=<0|1> python scripts/real_pilot_transfer.py <feats.pkl> <out.json>"""
import sys, json, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, "scripts")
import eval_v2 as E
from custody.detectors_v2 import Detector
F = pickle.load(open(sys.argv[1], "rb")); NC = E.NCOL
vec = lambda d: np.r_[d["ext"][:NC], d["pair"]] if not E.LOCAL else np.r_[d["ext"][:NC], d["pair"], d["loc"]]
keys = list(F); res = {}
if not E.LOCAL:
    det = Detector(joint_cols=list(range(NC))).fit(E.N, E.W, E.DEV)
    s = det.scores(np.array([np.r_[F[k]["src_ext"][:NC], np.zeros(5)] for k in keys]), np.array([vec(F[k]) for k in keys]))
    g = det.scores(E.N, E.W)
    for m in ("snapshot_conflict", "drift_only"):
        thr = float(np.percentile(g[m], 90)); sc = np.asarray(s[m])
        res[m] = dict(vision_p90=round(thr, 2), **{c: dict(n=int(sum(1 for k in keys if k[1] == c)), flagged=round(float(np.mean([sc[i] > thr for i, k in enumerate(keys) if k[1] == c])), 2), median=round(float(np.median([sc[i] for i, k in enumerate(keys) if k[1] == c])), 2)) for c in ("native", "ctrl", "copymove")})
else:
    forged = E.load_forged(("naive", "platform", "simctl"))
    Xp = np.array([forged[k] for k in forged if k[0] in ("naive", "platform")]); Xc = np.array([forged[k] for k in forged if k[0] == "simctl"])
    for name, neg in (("gbm_orig", E.W), ("gbm_ctrl", np.vstack([E.W, Xc]))):
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, class_weight="balanced", random_state=0).fit(np.vstack([neg, Xp]), np.r_[np.zeros(len(neg)), np.ones(len(Xp))])
        thr = float(np.percentile(clf.predict_proba(E.W)[:, 1], 90)); p = clf.predict_proba(np.array([vec(F[k]) for k in keys]))[:, 1]
        res[name] = dict(vision_p90=round(thr, 3), **{c: dict(n=int(sum(1 for k in keys if k[1] == c)), flagged=round(float(np.mean([p[i] > thr for i, k in enumerate(keys) if k[1] == c])), 2), median=round(float(np.median([p[i] for i, k in enumerate(keys) if k[1] == c])), 3)) for c in ("native", "ctrl", "copymove")})
json.dump(res, open(sys.argv[2], "w"), indent=1)
for m, v in res.items(): print(m, v)
