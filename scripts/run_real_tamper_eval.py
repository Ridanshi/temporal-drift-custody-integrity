import numpy as np

from custody.coevasion import CoEvasionTest
from custody.conflict import conflict_score
from custody.dfd_loader import real_tamper_deltas, real_tamper_joint_samples
from custody.drift import DriftModel
from custody.eval import evaluate
from custody.vision_loader import extract_genuine_deltas, extract_genuine_joint_samples

DIM = 4
TRANSFORM_TYPES = ("whatsapp", "youtube")


def build_real_tamper_eval_matrix(vision_root: str, dfd_root: str) -> dict:
    """Naive-tier (Section 3, Tier 1) validation using real deepfake video instead of a
    statistically simulated attacker. Genuine baselines are fit on the full real VISION
    genuine set, same as Table 3's single-split design, for direct comparability; the
    attack class here is real manipulated-vs-original deltas from the DFD dataset, inserted
    into the custody chain with no attempt to align metadata to the content, exactly the
    Tier-1 definition, just real instead of simulated. Heuristic and adaptive tiers still
    require an attacker who jointly forges metadata against this specific detector, which no
    available real dataset provides, so those tiers remain synthetic (Table 3) and are not
    reported here.
    """
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

    tamper_deltas = real_tamper_deltas(dfd_root)
    tamper_joint = real_tamper_joint_samples(dfd_root)
    n_tamper = len(tamper_deltas)
    n_tamper_joint = len(tamper_joint)

    result: dict = {}
    for transform_type in TRANSFORM_TYPES:
        genuine_deltas = deltas_by_type[transform_type]
        drift_threshold = drift_model.thresholds[transform_type]
        n_genuine = len(genuine_deltas)

        labels = np.concatenate([np.zeros(n_genuine), np.ones(n_tamper)])
        joint_labels = np.concatenate([np.zeros(len(genuine_joint)), np.ones(n_tamper_joint)])

        baseline_scores = np.concatenate(
            [
                np.array([conflict_score(d, np.zeros(DIM)) for d in genuine_deltas]),
                np.array([conflict_score(d, np.zeros(DIM)) for d in tamper_deltas]),
            ]
        )
        drift_scores_genuine = np.array([drift_model.score_edge(transform_type, d) for d in genuine_deltas])
        drift_scores_tamper = np.array([drift_model.score_edge(transform_type, d) for d in tamper_deltas])
        drift_scores = np.concatenate([drift_scores_genuine, drift_scores_tamper])

        coevasion_scores = np.concatenate(
            [
                np.array([coevasion_test.anomaly_score(j) for j in genuine_joint]),
                np.array([coevasion_test.anomaly_score(j) for j in tamper_joint]),
            ]
        )

        drift_excess = np.concatenate(
            [
                np.maximum(drift_scores_genuine - drift_threshold, 0.0),
                np.maximum(drift_scores_tamper - drift_threshold, 0.0),
            ]
        )
        coevasion_excess_for_full = np.concatenate(
            [
                np.array([coevasion_test.anomaly_score(genuine_joint[i % len(genuine_joint)]) for i in range(n_genuine)]),
                np.array([coevasion_test.anomaly_score(tamper_joint[i % n_tamper_joint]) for i in range(n_tamper)]),
            ]
        )
        full_scores = drift_excess + coevasion_excess_for_full

        result[transform_type] = {
            "snapshot_baseline": evaluate(baseline_scores, labels, baseline_threshold),
            "drift_only": evaluate(drift_scores, labels, drift_threshold),
            "coevasion_only": evaluate(coevasion_scores, joint_labels, 0.0),
            "full_system": evaluate(full_scores, labels, 0.0),
        }
    return result


if __name__ == "__main__":
    import json
    import sys

    vision_root = sys.argv[1] if len(sys.argv) > 1 else r"D:\DF paper\data\vision"
    dfd_root = sys.argv[2] if len(sys.argv) > 2 else r"D:\DF paper\data\dfd"
    print(json.dumps(build_real_tamper_eval_matrix(vision_root, dfd_root), indent=2))
