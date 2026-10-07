"""Proper post-hoc calibration (Platt scaling, isotonic regression) for the adaptive-tier
full-system score, fit on a held-out train split and evaluated on a disjoint test split,
avoiding the fit-and-eval-on-the-same-data problem that Section 7.4's raw min-max ECE has.
"""
import json

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from custody.attacks import adaptive_coevasion, heuristic_coevasion, naive_tamper
from custody.calibration import expected_calibration_error, scores_to_pseudo_probabilities
from custody.coevasion import CoEvasionTest
from custody.conflict import conflict_score
from custody.drift import DriftModel
from custody.vision_loader import extract_genuine_deltas, extract_genuine_joint_samples

DIM = 4
TRANSFORM_TYPES = ("whatsapp", "youtube")


def _attack_instances(delta_mean, joint_mean, coevasion_test, n, seed):
    deltas, joints = [], []
    for i in range(n):
        s = seed + i
        d = heuristic_coevasion(delta_mean, noise=0.02, seed=s)
        j = adaptive_coevasion(coevasion_test, joint_mean, n_steps=200, seed=s + 1)
        deltas.append(d)
        joints.append(j)
    return np.array(deltas), np.array(joints)


def run_calibration_experiment(vision_root: str, n_attack_instances: int = 60, seed: int = 0) -> dict:
    deltas_by_type = {t: extract_genuine_deltas(vision_root, t) for t in TRANSFORM_TYPES}
    joint_samples = extract_genuine_joint_samples(vision_root)

    drift_model = DriftModel(n_components=1)
    drift_model.fit(deltas_by_type, percentile=90.0)

    coevasion_test = CoEvasionTest()
    coevasion_test.fit(joint_samples, low_percentile=5.0, high_percentile=95.0)

    report = {}
    for transform_type in TRANSFORM_TYPES:
        genuine_deltas = deltas_by_type[transform_type]
        drift_threshold = drift_model.thresholds[transform_type]
        attack_deltas, attack_joints = _attack_instances(
            genuine_deltas.mean(axis=0), joint_samples.mean(axis=0), coevasion_test,
            n_attack_instances, seed=seed + 9000,
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

        train_scores, test_scores, train_labels, test_labels = train_test_split(
            full_scores, labels, test_size=0.4, random_state=seed, stratify=labels
        )

        raw_test_probs = scores_to_pseudo_probabilities(test_scores)
        raw_ece = expected_calibration_error(raw_test_probs, test_labels, n_bins=5)

        platt = LogisticRegression()
        platt.fit(train_scores.reshape(-1, 1), train_labels)
        platt_test_probs = platt.predict_proba(test_scores.reshape(-1, 1))[:, 1]
        platt_ece = expected_calibration_error(platt_test_probs, test_labels, n_bins=5)

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(train_scores, train_labels)
        iso_test_probs = iso.predict(test_scores)
        iso_ece = expected_calibration_error(iso_test_probs, test_labels, n_bins=5)

        report[transform_type] = {
            "n_train": len(train_scores),
            "n_test": len(test_scores),
            "raw_minmax_ece": round(float(raw_ece), 4),
            "platt_ece": round(float(platt_ece), 4),
            "isotonic_ece": round(float(iso_ece), 4),
        }
    return report


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\DF paper\data\vision"
    print(json.dumps(run_calibration_experiment(root), indent=2))
