"""Constrained-attacker sweep anchored on REAL tamper signatures (DFD, Celeb-DF).

An attacker starts from a real tamper delta/joint vector and spends a budget alpha in [0, 1]
to move it toward the genuine mean. alpha = 0 is the unmodified real tamper; alpha = 1 is the
unconstrained placement at the genuine mean used by the earlier adaptive tier. Two attackers:
  budget: moves both channels linearly toward the genuine mean (may overshoot into the
          too-perfect low tail of the co-evasion test);
  aware : knows the two-sided band, so its joint-channel move stops at the band centre.
Mahalanobis distance is a norm, so dist(mu + (1-a)(j-mu)) = (1-a) dist(j); the stopping point
for the aware attacker has the closed form a* = 1 - tau_mid / dist(j).
Also reports the real-tamper evaluation with the codec-hash dimension neutralised.
"""
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from custody.celebdf_loader import real_heuristic_tamper_deltas, real_heuristic_tamper_joint_samples
from custody.coevasion import CoEvasionTest
from custody.conflict import conflict_score
from custody.dfd_loader import real_tamper_deltas, real_tamper_joint_samples
from custody.drift import DriftModel
from custody.vision_loader import extract_genuine_deltas, extract_genuine_joint_samples

EDGES = ("whatsapp", "youtube")
ALPHAS = [round(0.1 * i, 1) for i in range(11)]


def run(vision_root, dfd_root, celeb_root, celeb_reenc):
    dbt = {e: extract_genuine_deltas(vision_root, e) for e in EDGES}
    gj = extract_genuine_joint_samples(vision_root)
    dm = DriftModel(n_components=1)
    dm.fit(dbt, percentile=90.0)
    ce = CoEvasionTest()
    ce.fit(gj, low_percentile=5.0, high_percentile=95.0)
    tau_mid = (ce.tau_low + ce.tau_high) / 2
    mu_j = gj.mean(axis=0)
    sets = {
        "dfd": (real_tamper_deltas(dfd_root), real_tamper_joint_samples(dfd_root)),
        "celebdf": (real_heuristic_tamper_deltas(celeb_root, celeb_reenc), real_heuristic_tamper_joint_samples(celeb_root, celeb_reenc)),
    }

    def scores(edge, gd, gjs, td, tj):
        n = len(gd)
        thr = dm.thresholds[edge]
        lab = np.r_[np.zeros(n), np.ones(len(td))]
        ei = EDGES.index(edge)
        gj_al = np.array([gjs[3 * i + 1 + ei] for i in range(n)])
        base = np.r_[[conflict_score(d, np.zeros(4)) for d in gd], [conflict_score(d, np.zeros(4)) for d in td]]
        drift = np.r_[[dm.score_edge(edge, d) for d in gd], [dm.score_edge(edge, d) for d in td]]
        cov_g = np.array([ce.anomaly_score(j) for j in gj_al])
        cov_t = np.array([ce.anomaly_score(j) for j in tj])
        full = np.maximum(drift - thr, 0) + np.r_[cov_g, cov_t]
        return {"snapshot_baseline": roc_auc_score(lab, base), "drift_only": roc_auc_score(lab, drift),
                "coevasion_only": roc_auc_score(lab, np.r_[cov_g, cov_t]), "full_system": roc_auc_score(lab, full)}

    out = {"sweep": {}, "codec_neutralised": {}, "alphas": ALPHAS, "tau_mid": tau_mid}
    for name, (td0, tj0) in sets.items():
        out["sweep"][name] = {}
        out["codec_neutralised"][name] = {}
        td_n, tj_n = td0.copy(), tj0.copy()
        td_n[:, 2] = 0.0
        tj_n[:, 2] = mu_j[2]
        for edge in EDGES:
            gd = dbt[edge]
            mu_d = gd.mean(axis=0)
            out["codec_neutralised"][name][edge] = {"as_run": scores(edge, gd, gj, td0, tj0), "codec_neutralised": scores(edge, gd, gj, td_n, tj_n)}
            out["sweep"][name][edge] = {}
            for att in ("budget", "aware"):
                out["sweep"][name][edge][att] = {}
                for a in ALPHAS:
                    td = (1 - a) * td_n + a * mu_d
                    d0 = np.array([ce.score(j) for j in tj_n])
                    a_eff = np.full(len(tj_n), a) if att == "budget" else np.minimum(a, np.where(d0 > tau_mid, 1 - tau_mid / np.maximum(d0, 1e-12), 0.0))
                    tj = mu_j + (1 - a_eff)[:, None] * (tj_n - mu_j)
                    out["sweep"][name][edge][att][str(a)] = scores(edge, gd, gj, td, tj)
    return out


if __name__ == "__main__":
    r = "E:/DF paper/data/"
    print(json.dumps(run(sys.argv[1] if len(sys.argv) > 1 else r + "vision", r + "dfd", r + "celebdf", r + "celebdf/manipulated_reencoded"), indent=1))
