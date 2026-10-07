import json
import sys

import numpy as np

from custody.attacks import adaptive_coevasion, heuristic_coevasion, naive_tamper
from custody.calibration import expected_calibration_error, scores_to_pseudo_probabilities
from custody.coevasion import CoEvasionTest
from custody.conflict import conflict_score
from custody.drift import DriftModel
from custody.eval import evaluate
from custody.ml_baseline import IsolationForestBaseline, OneClassSVMBaseline
from custody.pipeline import full_system_verdict
from custody.vision_loader import discover_triplets, extract_genuine_deltas, extract_genuine_joint_samples

DIM = 4
TRANSFORM_TYPES = ("whatsapp", "youtube")
TIERS = ("naive", "heuristic", "adaptive")


def _attack_instances(tier_name, delta_mean, joint_mean, coevasion_test, n, seed):
    deltas, joints = [], []
    for i in range(n):
        s = seed + i
        if tier_name == "naive":
            d = naive_tamper(delta_mean, magnitude=5.0, seed=s)
            j = naive_tamper(joint_mean, magnitude=5.0, seed=s + 1)
        elif tier_name == "heuristic":
            d = heuristic_coevasion(delta_mean, noise=0.3, seed=s)
            j = heuristic_coevasion(joint_mean, noise=0.3, seed=s + 1)
        elif tier_name == "adaptive":
            d = heuristic_coevasion(delta_mean, noise=0.02, seed=s)
            j = adaptive_coevasion(coevasion_test, joint_mean, n_steps=200, seed=s + 1)
        else:
            raise ValueError(tier_name)
        deltas.append(d)
        joints.append(j)
    return np.array(deltas), np.array(joints)


def _fit_common(vision_root, n_attack_instances, seed):
    deltas_by_type = {t: extract_genuine_deltas(vision_root, t) for t in TRANSFORM_TYPES}
    joint_samples = extract_genuine_joint_samples(vision_root)

    drift_model = DriftModel(n_components=1)
    drift_model.fit(deltas_by_type, percentile=90.0)

    coevasion_test = CoEvasionTest()
    coevasion_test.fit(joint_samples, low_percentile=5.0, high_percentile=95.0)

    return deltas_by_type, joint_samples, drift_model, coevasion_test


def ml_baseline_comparison(vision_root: str, n_attack_instances: int = 30, seed: int = 0) -> dict:
    """Task 17: Isolation Forest / One-Class SVM as additional baselines, fit on the same
    real genuine deltas as the drift model, scored on the same synthetic attack instances."""
    deltas_by_type, joint_samples, drift_model, coevasion_test = _fit_common(vision_root, n_attack_instances, seed)

    report = {}
    for transform_type in TRANSFORM_TYPES:
        genuine_deltas = deltas_by_type[transform_type]
        report[transform_type] = {}
        for tier_name in TIERS:
            attack_deltas, _ = _attack_instances(
                tier_name, genuine_deltas.mean(axis=0), joint_samples.mean(axis=0),
                coevasion_test, n_attack_instances, seed=seed + 2000 * (TIERS.index(tier_name) + 1),
            )
            labels = np.concatenate([np.zeros(len(genuine_deltas)), np.ones(len(attack_deltas))])

            iforest = IsolationForestBaseline(random_state=0)
            iforest.fit(genuine_deltas)
            iforest_scores = np.array(
                [iforest.anomaly_score(d) for d in np.concatenate([genuine_deltas, attack_deltas])]
            )
            iforest_threshold = float(np.percentile(
                [iforest.anomaly_score(d) for d in genuine_deltas], 90.0
            ))

            ocsvm = OneClassSVMBaseline(nu=0.2)
            ocsvm.fit(genuine_deltas)
            ocsvm_scores = np.array(
                [ocsvm.anomaly_score(d) for d in np.concatenate([genuine_deltas, attack_deltas])]
            )
            ocsvm_threshold = float(np.percentile(
                [ocsvm.anomaly_score(d) for d in genuine_deltas], 90.0
            ))

            report[transform_type][tier_name] = {
                "isolation_forest": evaluate(iforest_scores, labels, iforest_threshold),
                "one_class_svm": evaluate(ocsvm_scores, labels, ocsvm_threshold),
            }
    return report


def calibration_report(vision_root: str, n_attack_instances: int = 30, seed: int = 0) -> dict:
    """Task 18: Expected Calibration Error for the full-system score vs. the snapshot
    baseline score on the adaptive tier, the hardest and most informative tier."""
    deltas_by_type, joint_samples, drift_model, coevasion_test = _fit_common(vision_root, n_attack_instances, seed)

    report = {}
    for transform_type in TRANSFORM_TYPES:
        genuine_deltas = deltas_by_type[transform_type]
        drift_threshold = drift_model.thresholds[transform_type]
        attack_deltas, attack_joints = _attack_instances(
            "adaptive", genuine_deltas.mean(axis=0), joint_samples.mean(axis=0),
            coevasion_test, n_attack_instances, seed=seed + 9000,
        )
        n_genuine, n_attack = len(genuine_deltas), len(attack_deltas)
        labels = np.concatenate([np.zeros(n_genuine), np.ones(n_attack)])

        baseline_scores = np.array(
            [conflict_score(d, np.zeros(DIM)) for d in np.concatenate([genuine_deltas, attack_deltas])]
        )
        drift_excess = np.maximum(
            np.array([drift_model.score_edge(transform_type, d) for d in np.concatenate([genuine_deltas, attack_deltas])])
            - drift_threshold, 0.0,
        )
        coevasion_excess = np.array(
            [coevasion_test.anomaly_score(joint_samples[i % len(joint_samples)]) for i in range(n_genuine)]
            + [coevasion_test.anomaly_score(attack_joints[i]) for i in range(n_attack)]
        )
        full_scores = drift_excess + coevasion_excess

        report[transform_type] = {
            "snapshot_baseline_ece": expected_calibration_error(
                scores_to_pseudo_probabilities(baseline_scores), labels, n_bins=5
            ),
            "full_system_ece": expected_calibration_error(
                scores_to_pseudo_probabilities(full_scores), labels, n_bins=5
            ),
        }
    return report


def threshold_sensitivity_sweep(vision_root: str, n_attack_instances: int = 30, seed: int = 0) -> dict:
    """Task 19: sweep the calibration percentile and report AUC stability on the adaptive tier."""
    deltas_by_type, joint_samples, _, _ = _fit_common(vision_root, n_attack_instances, seed)

    report = {}
    for transform_type in TRANSFORM_TYPES:
        genuine_deltas = deltas_by_type[transform_type]
        report[transform_type] = {}
        for percentile in (80.0, 85.0, 90.0, 95.0, 99.0):
            drift_model = DriftModel(n_components=1)
            drift_model.fit({transform_type: genuine_deltas}, percentile=percentile)
            coevasion_test = CoEvasionTest()
            half = (100.0 - percentile) / 2.0
            coevasion_test.fit(joint_samples, low_percentile=half, high_percentile=100.0 - half)
            drift_threshold = drift_model.thresholds[transform_type]

            attack_deltas, attack_joints = _attack_instances(
                "adaptive", genuine_deltas.mean(axis=0), joint_samples.mean(axis=0),
                coevasion_test, n_attack_instances, seed=seed + 7000,
            )
            n_genuine, n_attack = len(genuine_deltas), len(attack_deltas)
            labels = np.concatenate([np.zeros(n_genuine), np.ones(n_attack)])

            drift_excess = np.maximum(
                np.array([drift_model.score_edge(transform_type, d) for d in np.concatenate([genuine_deltas, attack_deltas])])
                - drift_threshold, 0.0,
            )
            coevasion_excess = np.array(
                [coevasion_test.anomaly_score(joint_samples[i % len(joint_samples)]) for i in range(n_genuine)]
                + [coevasion_test.anomaly_score(attack_joints[i]) for i in range(n_attack)]
            )
            full_scores = drift_excess + coevasion_excess

            report[transform_type][str(percentile)] = evaluate(full_scores, labels, 0.0)["auc"]
    return report


def worked_case_study(vision_root: str) -> dict:
    """Task 20: one real triplet's actual feature vectors, scores, and verdict."""
    from custody.features import extract_intrinsic_features

    deltas_by_type, joint_samples, drift_model, coevasion_test = _fit_common(vision_root, 30, 0)
    triplet = discover_triplets(vision_root)[0]

    native_features = extract_intrinsic_features(triplet["native"])
    wa_features = extract_intrinsic_features(triplet["whatsapp"])
    delta = wa_features - native_features

    drift_score = drift_model.score_edge("whatsapp", delta)
    drift_threshold = drift_model.thresholds["whatsapp"]
    coevasion_score = coevasion_test.score(wa_features)
    coevasion_flag = coevasion_test.flag(wa_features)

    verdict = full_system_verdict(
        drift_flagged=drift_score > drift_threshold,
        path_break=False,
        coevasion_flag=coevasion_flag,
    )

    return {
        "device": triplet["device"],
        "scene": triplet["scene"],
        "native_features": native_features.tolist(),
        "whatsapp_features": wa_features.tolist(),
        "delta": delta.tolist(),
        "drift_score": drift_score,
        "drift_threshold": drift_threshold,
        "coevasion_score": coevasion_score,
        "coevasion_tau_low": coevasion_test.tau_low,
        "coevasion_tau_high": coevasion_test.tau_high,
        "coevasion_flag": coevasion_flag,
        "verdict_flagged": verdict.flagged,
        "verdict_reasons": verdict.reasons,
    }


def cross_transform_generalization(vision_root: str, n_attack_instances: int = 30, seed: int = 0) -> dict:
    """Task 21: fit the drift model on WhatsApp edges only, then score YouTube's genuine
    and attack deltas against that mismatched model, to test whether per-transform-type
    baselines are actually necessary or a single pooled model would do."""
    deltas_by_type, joint_samples, _, coevasion_test = _fit_common(vision_root, n_attack_instances, seed)

    whatsapp_deltas = deltas_by_type["whatsapp"]
    youtube_deltas = deltas_by_type["youtube"]

    matched_model = DriftModel(n_components=1)
    matched_model.fit({"youtube": youtube_deltas}, percentile=90.0)

    cross_model = DriftModel(n_components=1)
    cross_model.fit({"whatsapp": whatsapp_deltas}, percentile=90.0)

    attack_deltas, _ = _attack_instances(
        "adaptive", youtube_deltas.mean(axis=0), joint_samples.mean(axis=0),
        coevasion_test, n_attack_instances, seed=seed + 5000,
    )
    labels = np.concatenate([np.zeros(len(youtube_deltas)), np.ones(len(attack_deltas))])
    all_deltas = np.concatenate([youtube_deltas, attack_deltas])

    matched_scores = np.array([matched_model.score_edge("youtube", d) for d in all_deltas])
    matched_threshold = matched_model.thresholds["youtube"]

    cross_scores = np.array([cross_model.score_edge("whatsapp", d) for d in all_deltas])
    cross_threshold = cross_model.thresholds["whatsapp"]

    return {
        "matched_fit_on_youtube_auc": evaluate(matched_scores, labels, matched_threshold)["auc"],
        "cross_fit_whatsapp_scoring_youtube_auc": evaluate(cross_scores, labels, cross_threshold)["auc"],
    }


def run_all(vision_root: str, n_attack_instances: int = 30, seed: int = 0) -> dict:
    return {
        "ml_baseline_comparison": ml_baseline_comparison(vision_root, n_attack_instances, seed),
        "calibration": calibration_report(vision_root, n_attack_instances, seed),
        "threshold_sensitivity": threshold_sensitivity_sweep(vision_root, n_attack_instances, seed),
        "case_study": worked_case_study(vision_root),
        "cross_transform_generalization": cross_transform_generalization(vision_root, n_attack_instances, seed),
    }


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\DF paper\data\vision"
    print(json.dumps(run_all(root), indent=2))
