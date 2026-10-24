"""Secondary analyses on the real-forgery evaluation (run after eval_v2 data exist)."""
import json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "scripts")
import eval_v2 as E
from custody.detectors_v2 import Detector

OUT = {}
forged = E.load_forged()
tiers = [t for t in ("naive", "platform", "adaptive", "adaptive_drift_only", "s25", "s50") if any(k[0] == t for k in forged)]
dets = E.fold_detectors()
sc = E.collect(forged, tiers, dets)
M = E.METHODS
rng = np.random.default_rng(0)

# 1. per-edit AUC
OUT["per_op"] = {}
for tier in tiers:
    OUT["per_op"][tier] = {op: {m: E.auc_ci(sc, [(tier, op)], m, n_boot=2)[0] for m in M} for op in E.OPS if any((tier, op) in sc for _ in [0])}

# 2. paired device-cluster bootstrap of AUC differences
def paired_diff(tier, a, b, n_boot=2000):
    pg = [(tier, op) for op in E.OPS if (tier, op) in sc]
    def auc(m, devs):
        g = E.pooled(sc, "genuine", m, devs); p = np.concatenate([E.pooled(sc, x, m, devs) for x in pg])
        return roc_auc_score(np.r_[np.zeros(len(g)), np.ones(len(p))], np.r_[g, p])
    pt = auc(a, E.DEVS) - auc(b, E.DEVS)
    bs = []
    for _ in range(n_boot):
        d = list(rng.choice(E.DEVS, len(E.DEVS), replace=True))
        try:
            bs.append(auc(a, d) - auc(b, d))
        except Exception:
            pass
    bs = np.array(bs)
    return dict(diff=float(pt), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p_le0=float((bs <= 0).mean()))

OUT["paired"] = {}
pairs = [("full_system", "snapshot_conflict"), ("drift_only", "snapshot_conflict"), ("full_system", "drift_only"),
         ("coevasion_only", "mismatch_only"), ("drift_only", "ocsvm"), ("full_system", "ocsvm")]
for tier in tiers:
    OUT["paired"][tier] = {f"{a} - {b}": paired_diff(tier, a, b) for a, b in pairs}

# 3. channel ablations (platform + naive + adaptive tiers pooled by tier)
def run_variant(drift_cols=None, joint_cols=None, label=""):
    d = {dv: Detector(drift_cols=drift_cols, joint_cols=joint_cols).fit(E.N[E.DEV != dv], E.W[E.DEV != dv], E.DEV[E.DEV != dv]) for dv in E.DEVS}
    s = E.collect(forged, tiers, d)
    return {tier: {m: E.auc_ci(s, [(tier, op) for op in E.OPS if (tier, op) in s], m, n_boot=2)[0] for m in ("drift_only", "coevasion_only", "full_system")} for tier in tiers}

nf = E.NCOL
allc = [j for j in range(nf + E.NP) if j != 3]
OUT["ablation"] = {
    "all_features": run_variant(None, list(range(nf))),
    "drop_pair_features": run_variant([j for j in allc if j < nf], list(range(nf))),
    "pair_features_only": run_variant([j for j in allc if j >= nf], list(range(nf))),
    "declared_only": run_variant([0, 1, 2], [0, 1, 2, 3]),
    "measured_only": run_variant([j for j in allc if j >= 4], list(range(4, nf))),
}

# 4. fidelity: real WhatsApp vs untampered simulated WhatsApp, device-grouped logistic AUC
sim = np.array([forged[("simctl", i, None)] for i in range(len(E.T))])
X = np.vstack([E.W, sim]); y = np.r_[np.zeros(len(E.W)), np.ones(len(sim))]; g = np.r_[E.DEV, E.DEV]
aucs = []
pred = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, g):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.3)).fit(X[tr], y[tr])
    pred[te] = clf.predict_proba(X[te])[:, 1]
OUT["fidelity_real_vs_sim_auc"] = float(roc_auc_score(y, pred))

# 5. false declaration: real YouTube edges scored by the WhatsApp detector, and WA scored by YT detector
fd = {}
for claim, true_arr, other_arr, name in (("whatsapp", E.Y, E.W, "yt_claimed_wa"), ("youtube", E.W, E.Y, "wa_claimed_yt")):
    s_true, s_false = [], []
    for dv in E.DEVS:
        te = E.DEV == dv
        det = Detector(joint_cols=list(range(nf))).fit(E.N[~te], other_arr[~te], E.DEV[~te])
        ss_true = det.scores(E.N[te], other_arr[te]); ss_false = det.scores(E.N[te], true_arr[te])
        s_true.append(ss_true); s_false.append(ss_false)
    fd[name] = {m: dict(auc=float(roc_auc_score(np.r_[np.zeros(86), np.ones(86)], np.r_[np.concatenate([x[m] for x in s_true]), np.concatenate([x[m] for x in s_false])])),
                        flag_true=float(np.mean(np.concatenate([x[m] for x in s_true]) > 0)), flag_false=float(np.mean(np.concatenate([x[m] for x in s_false]) > 0)))
                for m in M}
OUT["false_declaration"] = fd
json.dump(OUT, open(f"scripts/analysis_v2_{E.FEATSET}{'_pair' if E.PAIR else ''}.json", "w"), indent=1)
print(json.dumps({k: OUT[k] for k in ("fidelity_real_vs_sim_auc",)}, indent=1))
for tier in tiers:
    print(tier, "paired:")
    for k, v in OUT["paired"][tier].items(): print("  %-34s %+.3f [%+.3f,%+.3f] p<=0: %.3f" % (k, v["diff"], v["lo"], v["hi"], v["p_le0"]))
print("ablation:"); 
for k, v in OUT["ablation"].items(): print(" ", k, {t: {m[:5]: round(x, 3) for m, x in d.items()} for t, d in v.items()})
print("false declaration:")
for k, v in fd.items(): print(" ", k, {m: (round(x["auc"], 3), round(x["flag_true"], 2), round(x["flag_false"], 2)) for m, x in v.items() if m in ("drift_only", "full_system", "snapshot_conflict")})
