import numpy as np
from sklearn.metrics import f1_score

from custody.attacks import adaptive_coevasion, heuristic_coevasion, naive_tamper
from custody.coevasion import CoEvasionTest
from custody.conflict import conflict_score
from custody.drift import DriftModel
from custody.eval import evaluate
from custody.vision_loader import deltas_for_triplets, discover_triplets, joint_samples_for_triplets

DIM = 4
TRANSFORM_TYPES = ("whatsapp", "youtube")
TIERS = ("naive", "heuristic", "adaptive")
METHODS = ("snapshot_baseline", "drift_only", "coevasion_only", "full_system")


def _generate_tier_instances(
    tier_name: str,
    delta_mean: np.ndarray,
    joint_mean: np.ndarray,
    coevasion_test: CoEvasionTest,
    n_attack_instances: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    deltas, joints = [], []
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


def build_held_out_eval_matrix(vision_root: str, n_attack_instances: int = 30, seed: int = 0) -> dict:
    """Leave-one-device-out cross-validation over the real VISION subset.

    For each held-out device, every model (drift GMM, co-evasion joint distribution, baseline
    threshold) is fit only on the *other* devices' genuine triplets, then scored against the
    held-out device's genuine triplets plus synthetic attack instances anchored at the
    training-fold genuine mean. No genuine sample is ever scored by a model that was fit on it,
    which is what Section 7.2.3/8.5 flags as missing from the single-split evaluation in
    Table 3. AUC is computed per fold (raw scores from differently-fit GMMs across folds are
    not on a guaranteed-comparable scale) and reported as mean +/- std across folds; F1 pools
    binary predictions across folds, which is scale-safe.
    """
    triplets = discover_triplets(vision_root)
    devices = sorted({t["device"] for t in triplets})

    fold_aucs: dict = {
        tt: {m: {tier: [] for tier in TIERS} for m in METHODS} for tt in TRANSFORM_TYPES
    }
    pooled_preds: dict = {
        tt: {m: {tier: {"preds": [], "labels": []} for tier in TIERS} for m in METHODS}
        for tt in TRANSFORM_TYPES
    }

    for fold_idx, held_out_device in enumerate(devices):
        train_triplets = [t for t in triplets if t["device"] != held_out_device]
        test_triplets = [t for t in triplets if t["device"] == held_out_device]

        train_deltas_by_type = {
            tt: deltas_for_triplets(train_triplets, vision_root, tt) for tt in TRANSFORM_TYPES
        }
        train_joint = joint_samples_for_triplets(train_triplets, vision_root)

        drift_model = DriftModel(n_components=1)
        drift_model.fit(train_deltas_by_type, percentile=90.0)

        coevasion_test = CoEvasionTest()
        coevasion_test.fit(train_joint, low_percentile=5.0, high_percentile=95.0)

        baseline_all = np.concatenate(
            [np.array([conflict_score(d, np.zeros(DIM)) for d in deltas]) for deltas in train_deltas_by_type.values()]
        )
        baseline_threshold = float(np.percentile(baseline_all, 90.0))

        test_deltas_by_type = {
            tt: deltas_for_triplets(test_triplets, vision_root, tt) for tt in TRANSFORM_TYPES
        }
        test_joint = joint_samples_for_triplets(test_triplets, vision_root)
        n_test_joint = len(test_joint)

        for tt in TRANSFORM_TYPES:
            drift_threshold = drift_model.thresholds[tt]
            test_deltas = test_deltas_by_type[tt]
            n_test_genuine = len(test_deltas)

            for tier_name in TIERS:
                fold_seed = seed + 1000 * (TIERS.index(tier_name) + 1) + 100 * fold_idx
                attack_deltas, attack_joints = _generate_tier_instances(
                    tier_name,
                    train_deltas_by_type[tt].mean(axis=0),
                    train_joint.mean(axis=0),
                    coevasion_test,
                    n_attack_instances,
                    fold_seed,
                )

                labels = np.concatenate([np.zeros(n_test_genuine), np.ones(n_attack_instances)])
                joint_labels = np.concatenate([np.zeros(n_test_joint), np.ones(n_attack_instances)])

                baseline_scores = np.concatenate(
                    [
                        np.array([conflict_score(d, np.zeros(DIM)) for d in test_deltas]),
                        np.array([conflict_score(d, np.zeros(DIM)) for d in attack_deltas]),
                    ]
                )
                drift_scores_test = np.array([drift_model.score_edge(tt, d) for d in test_deltas])
                drift_scores_attack = np.array([drift_model.score_edge(tt, d) for d in attack_deltas])
                drift_scores = np.concatenate([drift_scores_test, drift_scores_attack])

                coevasion_scores = np.concatenate(
                    [
                        np.array([coevasion_test.anomaly_score(j) for j in test_joint]),
                        np.array([coevasion_test.anomaly_score(j) for j in attack_joints]),
                    ]
                )

                drift_excess = np.concatenate(
                    [
                        np.maximum(drift_scores_test - drift_threshold, 0.0),
                        np.maximum(drift_scores_attack - drift_threshold, 0.0),
                    ]
                )
                coevasion_excess_for_full = np.concatenate(
                    [
                        np.array([coevasion_test.anomaly_score(test_joint[i % n_test_joint]) for i in range(n_test_genuine)]),
                        np.array([coevasion_test.anomaly_score(attack_joints[i]) for i in range(n_attack_instances)]),
                    ]
                )
                full_scores = drift_excess + coevasion_excess_for_full

                per_method = {
                    "snapshot_baseline": (baseline_scores, labels, baseline_threshold),
                    "drift_only": (drift_scores, labels, drift_threshold),
                    "coevasion_only": (coevasion_scores, joint_labels, 0.0),
                    "full_system": (full_scores, labels, 0.0),
                }
                for method, (scores, lbls, threshold) in per_method.items():
                    fold_result = evaluate(scores, lbls, threshold)
                    fold_aucs[tt][method][tier_name].append(fold_result["auc"])
                    preds = (scores > threshold).astype(int)
                    pooled_preds[tt][method][tier_name]["preds"].append(preds)
                    pooled_preds[tt][method][tier_name]["labels"].append(lbls)

    result: dict = {}
    for tt in TRANSFORM_TYPES:
        result[tt] = {}
        for method in METHODS:
            result[tt][method] = {}
            for tier_name in TIERS:
                aucs = np.array(fold_aucs[tt][method][tier_name])
                preds = np.concatenate(pooled_preds[tt][method][tier_name]["preds"])
                labels = np.concatenate(pooled_preds[tt][method][tier_name]["labels"])

                result[tt][method][tier_name] = {
                    "auc_mean": float(aucs.mean()),
                    "auc_std": float(aucs.std()),
                    "auc_per_fold": aucs.tolist(),
                    "f1_pooled": float(f1_score(labels, preds)),
                    "n_folds": len(aucs),
                }
    return result


if __name__ == "__main__":
    import json
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\DF paper\data\vision"
    print(json.dumps(build_held_out_eval_matrix(root), indent=2))
