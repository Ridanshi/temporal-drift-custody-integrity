"""Recalibrated test on REAL current-WhatsApp copies: leave-one-clip-out, edit-only (copymove vs the clip's own ctrl).
Each held-out clip is scored by a model fitted only on the OTHER clips' real copies.
  python scripts/real_pilot_eval.py work/real_pilot_feats.pkl work/degraded/real_pilot_eval.json"""
import sys, json, pickle
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

F = pickle.load(open(sys.argv[1], "rb"))
vec = lambda d: np.r_[d["ext"][:12], d["pair"], d["loc"]]
X = {k: vec(v) for k, v in F.items()}
import os
strict = set(json.load(open("work/real_pilot_strict_clips.json"))) if os.environ.get("STRICT") == "1" else None
clips = sorted({c for c, o in X if (c, "ctrl") in X and (c, "copymove") in X and (strict is None or c in strict)})
print(len(clips), "clips with ctrl and copymove:", clips)
COLS = {"all25": list(range(25)), "edit13 (pair+local)": list(range(12, 25)), "local8": list(range(17, 25)), "pair5": list(range(12, 17))}
rng = np.random.default_rng(0)

def auc_from(sc, idx):
    a = np.array([sc[c][0] for c in idx]); b = np.array([sc[c][1] for c in idx])  # (ctrl, copymove)
    return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])

out = {}
for cname, cols in COLS.items():
    for model in ("mahalanobis", "logistic"):
        sc = {}
        for c in clips:
            rest = [r for r in clips if r != c]
            if model == "mahalanobis":
                G = np.array([X[(r, o)][cols] for r in rest for o in ("native", "ctrl") if (r, o) in X])
                mu = G.mean(0); sd = G.std(0) + 1e-6
                lw = LedoitWolf().fit((G - mu) / sd); P = lw.precision_
                s = lambda x: float(np.sqrt(max(((x[cols] - mu) / sd) @ P @ ((x[cols] - mu) / sd), 0)))
                sc[c] = (s(X[(c, "ctrl")]), s(X[(c, "copymove")]))
            else:
                Xn = np.array([X[(r, "ctrl")][cols] for r in rest]); Xp = np.array([X[(r, "copymove")][cols] for r in rest])
                sca = StandardScaler().fit(np.vstack([Xn, Xp]))
                clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000).fit(sca.transform(np.vstack([Xn, Xp])), np.r_[np.zeros(len(Xn)), np.ones(len(Xp))])
                p = lambda x: float(clf.predict_proba(sca.transform(x[cols][None]))[0, 1])
                sc[c] = (p(X[(c, "ctrl")]), p(X[(c, "copymove")]))
        pt = auc_from(sc, clips); bs = [auc_from(sc, list(rng.choice(clips, len(clips)))) for _ in range(2000)]
        wins = sum(1 for c in clips if sc[c][1] > sc[c][0])
        out[f"{model} | {cname}"] = dict(auc=round(pt, 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], clips_copymove_higher=f"{wins}/{len(clips)}", sign_test_p=round(binomtest(wins, len(clips), 0.5, alternative="greater").pvalue, 3))
        print(f"{model:12s} {cname:22s} AUC {pt:.2f} [{out[f'{model} | {cname}']['ci'][0]:.2f}, {out[f'{model} | {cname}']['ci'][1]:.2f}]  copymove>ctrl in {wins}/{len(clips)} clips (sign test p={out[f'{model} | {cname}']['sign_test_p']})")
json.dump(dict(clips=clips, results=out), open(sys.argv[2], "w"), indent=1)
