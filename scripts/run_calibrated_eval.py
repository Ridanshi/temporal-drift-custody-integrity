import numpy as np

from custody.attacks import adaptive_coevasion, heuristic_coevasion, naive_tamper
from custody.coevasion import CoEvasionTest
from custody.conflict import conflict_score
from custody.drift import DriftModel
from custody.eval import run_eval_matrix
from custody.synthetic import generate_genuine_deltas, generate_joint_samples

DIM = 4
TRANSFORM_TYPE = "re_encode"


def _generate_tier_instances(
    tier_name: str,
    delta_mean: np.ndarray,
    joint_mean: np.ndarray,
    coevasion_test: CoEvasionTest,
    n_attack_instances: int,
    seed: int,
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
        deltas.append(d)
        joints.append(j)
    return np.array(deltas), np.array(joints)


def build_calibrated_eval_matrix(
    n_genuine: int = 200, n_attack_instances: int = 100, seed: int = 0
) -> dict:
    train_deltas = generate_genuine_deltas(TRANSFORM_TYPE, n=n_genuine, dim=DIM, seed=seed)
    train_joint = generate_joint_samples(n=n_genuine, dim=DIM, seed=seed + 1)

    drift_model = DriftModel(n_components=1)
    drift_model.fit({TRANSFORM_TYPE: train_deltas}, percentile=95.0)

    coevasion_test = CoEvasionTest()
    coevasion_test.fit(train_joint, low_percentile=1.0, high_percentile=99.0)

    baseline_train_scores = np.array([conflict_score(d, np.zeros(DIM)) for d in train_deltas])
    baseline_threshold = float(np.percentile(baseline_train_scores, 95.0))
    drift_threshold = drift_model.thresholds[TRANSFORM_TYPE]

    test_deltas = generate_genuine_deltas(TRANSFORM_TYPE, n=n_genuine, dim=DIM, seed=seed + 500)
    test_joint = generate_joint_samples(n=n_genuine, dim=DIM, seed=seed + 501)

    tiers = ("naive", "heuristic", "adaptive")
    per_tier = {}
    for tier_name in tiers:
        attack_deltas, attack_joints = _generate_tier_instances(
            tier_name,
            train_deltas.mean(axis=0),
            train_joint.mean(axis=0),
            coevasion_test,
            n_attack_instances,
            seed=seed + 1000 * (tiers.index(tier_name) + 1),
        )

        labels = np.concatenate([np.zeros(n_genuine), np.ones(n_attack_instances)])

        baseline_scores = np.concatenate(
            [
                baseline_scores_for(test_deltas),
                baseline_scores_for(attack_deltas),
            ]
        )
        drift_scores = np.concatenate(
            [
                drift_scores_for(drift_model, test_deltas),
                drift_scores_for(drift_model, attack_deltas),
            ]
        )
        coevasion_scores = np.concatenate(
            [
                coevasion_anomaly_for(coevasion_test, test_joint),
                coevasion_anomaly_for(coevasion_test, attack_joints),
            ]
        )
        drift_excess = np.concatenate(
            [
                np.maximum(drift_scores_for(drift_model, test_deltas) - drift_threshold, 0.0),
                np.maximum(drift_scores_for(drift_model, attack_deltas) - drift_threshold, 0.0),
            ]
        )
        full_system_scores = drift_excess + coevasion_scores

        per_tier[tier_name] = {
            "labels": labels,
            "baseline": baseline_scores,
            "drift": drift_scores,
            "coevasion": coevasion_scores,
            "full_system": full_system_scores,
        }

    methods = {
        "snapshot_baseline": ("baseline", baseline_threshold),
        "drift_only": ("drift", drift_threshold),
        "coevasion_only": ("coevasion", 0.0),
        "full_system": ("full_system", 0.0),
    }

    result: dict = {}
    for method_name, (score_key, threshold) in methods.items():
        attack_tiers = {
            tier_name: (per_tier[tier_name][score_key], per_tier[tier_name]["labels"])
            for tier_name in tiers
        }
        result[method_name] = run_eval_matrix(
            {method_name: lambda s: s}, attack_tiers, threshold=threshold
        )[method_name]

    return result


def baseline_scores_for(deltas: np.ndarray) -> np.ndarray:
    return np.array([conflict_score(d, np.zeros(DIM)) for d in deltas])


def drift_scores_for(drift_model: DriftModel, deltas: np.ndarray) -> np.ndarray:
    return np.array([drift_model.score_edge(TRANSFORM_TYPE, d) for d in deltas])


def coevasion_anomaly_for(coevasion_test: CoEvasionTest, joints: np.ndarray) -> np.ndarray:
    return np.array([coevasion_test.anomaly_score(j) for j in joints])


if __name__ == "__main__":
    import json

    print(json.dumps(build_calibrated_eval_matrix(), indent=2))
