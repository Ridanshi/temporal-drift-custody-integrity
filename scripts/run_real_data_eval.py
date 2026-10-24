import numpy as np

from custody.attacks import adaptive_coevasion, heuristic_coevasion, naive_tamper
from custody.coevasion import CoEvasionTest
from custody.conflict import conflict_score
from custody.drift import DriftModel
from custody.eval import evaluate
from custody.vision_loader import extract_genuine_deltas, extract_genuine_joint_samples

DIM = 4
TRANSFORM_TYPES = ("whatsapp", "youtube")


def _generate_tier_instances(
    tier_name: str,
    delta_mean: np.ndarray,
    joint_mean: np.ndarray,
    coevasion_test: CoEvasionTest,
    n_attack_instances: int,
    seed: int,
    dmask=None,
    jmask=None,
) -> tuple[np.ndarray, np.ndarray]:
    deltas = []
    joints = []
    for i in range(n_attack_instances):
        instance_seed = seed + i
        if tier_name == "naive":
            d = naive_tamper(delta_mean, magnitude=5.0, seed=instance_seed)
            j = naive_tamper(joint_mean, magnitude=5.0, seed=instance_seed + 1)
        elif tier_name == "heuristic":
            d = heuristic_coevasion(delta_mean, noise=0.3, seed=instance_seed)
            j = heuristic_coevasion(joint_mean, noise=0.3, seed=instance_seed + 1)
        elif tier_name == "adaptive":
            d = heuristic_coevasion(delta_mean, noise=0.02, seed=instance_seed)
            j = adaptive_coevasion(coevasion_test, joint_mean, n_steps=200, seed=instance_seed + 1)
        else:
            raise ValueError(f"unknown tier: {tier_name}")
        if dmask is not None:
            d = np.where(dmask, d, delta_mean)
        if jmask is not None:
            j = np.where(jmask, j, joint_mean)
        deltas.append(d)
        joints.append(j)
    return np.array(deltas), np.array(joints)


def build_real_data_eval_matrix(
    vision_root: str, n_attack_instances: int = 30, seed: int = 0, return_raw: bool = False
) -> dict:
    """Fits genuine baselines from real VISION-derived deltas/joint samples (small N: one
    sample per available triplet), then evaluates against synthetic attack instances anchored
    at the real genuine mean, since no real tampered-video corpus is available. Genuine
    "test" samples reuse the fitting set itself (N too small to hold out a separate real
    test split) -- this is an initial real-data proof of concept, not a held-out evaluation.
    """
    deltas_by_type = {
        t: extract_genuine_deltas(vision_root, t) for t in TRANSFORM_TYPES
    }
    joint_samples = extract_genuine_joint_samples(vision_root)

    drift_model = DriftModel(n_components=1)
    drift_model.fit(deltas_by_type, percentile=90.0)

    coevasion_test = CoEvasionTest()
    coevasion_test.fit(joint_samples, low_percentile=5.0, high_percentile=95.0)

    baseline_all = np.concatenate(
        [np.array([conflict_score(d, np.zeros(DIM)) for d in deltas]) for deltas in deltas_by_type.values()]
    )
    baseline_threshold = float(np.percentile(baseline_all, 90.0))

    tiers = ("naive", "heuristic", "adaptive")
    result: dict = {}

    for transform_type in TRANSFORM_TYPES:
        genuine_deltas = deltas_by_type[transform_type]
        genuine_joint = joint_samples
        drift_threshold = drift_model.thresholds[transform_type]

        result[transform_type] = {
            "snapshot_baseline": {},
            "drift_only": {},
            "coevasion_only": {},
            "full_system": {},
        }

        for tier_name in tiers:
            attack_deltas, attack_joints = _generate_tier_instances(
                tier_name,
                genuine_deltas.mean(axis=0),
                genuine_joint.mean(axis=0),
                coevasion_test,
                n_attack_instances,
                seed=seed + 1000 * (tiers.index(tier_name) + 1),
                dmask=genuine_deltas.std(axis=0) > 1e-9,
                jmask=genuine_joint.std(axis=0) > 1e-9,
            )

            n_genuine = len(genuine_deltas)
            labels = np.concatenate([np.zeros(n_genuine), np.ones(n_attack_instances)])

            baseline_scores = np.concatenate(
                [
                    np.array([conflict_score(d, np.zeros(DIM)) for d in genuine_deltas]),
                    np.array([conflict_score(d, np.zeros(DIM)) for d in attack_deltas]),
                ]
            )
            drift_scores = np.concatenate(
                [
                    np.array([drift_model.score_edge(transform_type, d) for d in genuine_deltas]),
                    np.array([drift_model.score_edge(transform_type, d) for d in attack_deltas]),
                ]
            )

            n_genuine_joint = len(genuine_joint)
            joint_labels = np.concatenate([np.zeros(n_genuine_joint), np.ones(n_attack_instances)])
            coevasion_scores = np.concatenate(
                [
                    np.array([coevasion_test.anomaly_score(j) for j in genuine_joint]),
                    np.array([coevasion_test.anomaly_score(j) for j in attack_joints]),
                ]
            )

            drift_excess = np.concatenate(
                [
                    np.maximum(
                        np.array([drift_model.score_edge(transform_type, d) for d in genuine_deltas])
                        - drift_threshold,
                        0.0,
                    ),
                    np.maximum(
                        np.array([drift_model.score_edge(transform_type, d) for d in attack_deltas])
                        - drift_threshold,
                        0.0,
                    ),
                ]
            )
            coevasion_excess_for_full = np.concatenate(
                [
                    np.array([coevasion_test.anomaly_score(genuine_joint[i % n_genuine_joint]) for i in range(n_genuine)]),
                    np.array([coevasion_test.anomaly_score(attack_joints[i]) for i in range(n_attack_instances)]),
                ]
            )
            full_scores = drift_excess + coevasion_excess_for_full

            result[transform_type]["snapshot_baseline"][tier_name] = evaluate(
                baseline_scores, labels, baseline_threshold
            )
            result[transform_type]["drift_only"][tier_name] = evaluate(drift_scores, labels, drift_threshold)
            result[transform_type]["coevasion_only"][tier_name] = evaluate(
                coevasion_scores, joint_labels, 0.0
            )
            result[transform_type]["full_system"][tier_name] = evaluate(full_scores, labels, 0.0)

            if return_raw:
                result[transform_type].setdefault("_raw", {})[tier_name] = {
                    "snapshot_baseline": (baseline_scores, labels, baseline_threshold),
                    "full_system": (full_scores, labels, 0.0),
                }

    return result


def paired_correctness(scores: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    preds = (scores > threshold).astype(int)
    return preds == labels


if __name__ == "__main__":
    import json
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\DF paper\data\vision"
    print(json.dumps(build_real_data_eval_matrix(root), indent=2))
