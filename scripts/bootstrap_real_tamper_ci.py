"""Bootstrap confidence intervals for the real-data Tier-1 (DFD) and Tier-2 (Celeb-DF)
validations reported in Sections 7.2.5 and 7.2.6. Both are single-split evaluations
(genuine class n=8, attack class n=30 or 40); this is a standard two-sample percentile
bootstrap on AUC, resampling the genuine and attack score pools independently with
replacement, distinct from Section 7.2.4's cluster (fold-level) bootstrap since there is
no fold structure here."""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from custody.coevasion import CoEvasionTest
from custody.conflict import conflict_score
from custody.dfd_loader import real_tamper_deltas, real_tamper_joint_samples
from custody.celebdf_loader import real_heuristic_tamper_deltas, real_heuristic_tamper_joint_samples
from custody.drift import DriftModel
from custody.vision_loader import extract_genuine_deltas, extract_genuine_joint_samples

DIM = 4
TRANSFORM_TYPES = ("whatsapp", "youtube")
RNG = np.random.default_rng(0)
N_BOOT = 10000


def bootstrap_auc_ci(genuine_scores, attack_scores, n_boot=N_BOOT, alpha=0.05):
    genuine_scores = np.asarray(genuine_scores)
    attack_scores = np.asarray(attack_scores)
    boot_aucs = []
    for _ in range(n_boot):
        g = RNG.choice(genuine_scores, size=len(genuine_scores), replace=True)
        a = RNG.choice(attack_scores, size=len(attack_scores), replace=True)
        labels = np.concatenate([np.zeros(len(g)), np.ones(len(a))])
        scores = np.concatenate([g, a])
        try:
            boot_aucs.append(roc_auc_score(labels, scores))
        except ValueError:
            continue
    boot_aucs = np.array(boot_aucs)
    lo, hi = np.percentile(boot_aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(np.mean(boot_aucs)), int(len(boot_aucs))


def scores_for_pass(vision_root, tamper_deltas, tamper_joint):
    deltas_by_type = {t: extract_genuine_deltas(vision_root, t) for t in TRANSFORM_TYPES}
    genuine_joint = extract_genuine_joint_samples(vision_root)

    drift_model = DriftModel(n_components=1)
    drift_model.fit(deltas_by_type, percentile=90.0)

    coevasion_test = CoEvasionTest()
    coevasion_test.fit(genuine_joint, low_percentile=5.0, high_percentile=95.0)

    baseline_all = np.concatenate(
        [np.array([conflict_score(d, np.zeros(DIM)) for d in deltas]) for deltas in deltas_by_type.values()]
    )
    baseline_threshold = float(np.percentile(baseline_all, 90.0))

    result = {}
    for tt in TRANSFORM_TYPES:
        genuine_deltas = deltas_by_type[tt]
        drift_threshold = drift_model.thresholds[tt]

        genuine_baseline = np.array([conflict_score(d, np.zeros(DIM)) for d in genuine_deltas])
        attack_baseline = np.array([conflict_score(d, np.zeros(DIM)) for d in tamper_deltas])

        genuine_drift = np.array([drift_model.score_edge(tt, d) for d in genuine_deltas])
        attack_drift = np.array([drift_model.score_edge(tt, d) for d in tamper_deltas])

        genuine_coev = np.array([coevasion_test.anomaly_score(j) for j in genuine_joint])
        attack_coev = np.array([coevasion_test.anomaly_score(j) for j in tamper_joint])

        genuine_drift_excess = np.maximum(genuine_drift - drift_threshold, 0.0)
        attack_drift_excess = np.maximum(attack_drift - drift_threshold, 0.0)
        n_genuine, n_attack = len(genuine_deltas), len(tamper_deltas)
        genuine_coev_for_full = np.array(
            [coevasion_test.anomaly_score(genuine_joint[i % len(genuine_joint)]) for i in range(n_genuine)]
        )
        attack_coev_for_full = np.array(
            [coevasion_test.anomaly_score(tamper_joint[i % len(tamper_joint)]) for i in range(n_attack)]
        )
        genuine_full = genuine_drift_excess + genuine_coev_for_full
        attack_full = attack_drift_excess + attack_coev_for_full

        result[tt] = {
            "snapshot_baseline": (genuine_baseline, attack_baseline),
            "drift_only": (genuine_drift, attack_drift),
            "coevasion_only": (genuine_coev, attack_coev),
            "full_system": (genuine_full, attack_full),
        }
    return result


def ci_report(scores_by_type):
    out = {}
    for tt, methods in scores_by_type.items():
        out[tt] = {}
        for method, (g, a) in methods.items():
            lo, hi, mean, n = bootstrap_auc_ci(g, a)
            out[tt][method] = {"ci_95_low": round(lo, 3), "ci_95_high": round(hi, 3), "bootstrap_mean": round(mean, 3)}
    return out


if __name__ == "__main__":
    vision_root = r"D:\DF paper\data\vision"
    dfd_root = r"D:\DF paper\data\dfd"
    celebdf_root = r"D:\DF paper\data\celebdf"
    reencode_dir = r"D:\DF paper\data\celebdf\manipulated_reencoded"

    dfd_scores = scores_for_pass(vision_root, real_tamper_deltas(dfd_root), real_tamper_joint_samples(dfd_root))
    celebdf_scores = scores_for_pass(
        vision_root,
        real_heuristic_tamper_deltas(celebdf_root, reencode_dir),
        real_heuristic_tamper_joint_samples(celebdf_root, reencode_dir),
    )

    report = {
        "dfd_naive_tier": ci_report(dfd_scores),
        "celebdf_heuristic_tier": ci_report(celebdf_scores),
    }
    print(json.dumps(report, indent=2))
