import numpy as np

from custody.attacks import adaptive_coevasion, heuristic_coevasion, naive_tamper
from custody.coevasion import CoEvasionTest
from custody.conflict import conflict_score
from custody.drift import DriftModel
from custody.eval import run_eval_matrix
from custody.synthetic import generate_genuine_deltas, generate_joint_samples

DIM = 4
TRANSFORM_TYPE = "re_encode"
N_GENUINE = 200
N_ATTACK = 200


def _build_scores(seed: int) -> dict:
    genuine_deltas = generate_genuine_deltas(TRANSFORM_TYPE, n=N_GENUINE, dim=DIM, seed=seed)
    genuine_joint = generate_joint_samples(n=N_GENUINE, dim=DIM, seed=seed + 1)

    drift_model = DriftModel(n_components=1)
    drift_model.fit({TRANSFORM_TYPE: genuine_deltas})

    coevasion_test = CoEvasionTest()
    coevasion_test.fit(genuine_joint)

    tiers = {}
    for tier_name, attack_fn in (
        ("naive", lambda: naive_tamper(genuine_deltas.mean(axis=0), magnitude=5.0, seed=seed)),
        ("heuristic", lambda: heuristic_coevasion(genuine_joint.mean(axis=0), noise=0.3, seed=seed)),
        ("adaptive", lambda: adaptive_coevasion(coevasion_test, genuine_joint.mean(axis=0), seed=seed)),
    ):
        attack_sample = attack_fn()

        drift_scores = np.concatenate(
            [
                np.array([drift_model.score_edge(TRANSFORM_TYPE, d) for d in genuine_deltas]),
                np.array([drift_model.score_edge(TRANSFORM_TYPE, attack_sample[:DIM])] * N_ATTACK),
            ]
        )
        coevasion_scores = np.concatenate(
            [
                np.array([coevasion_test.score(s) for s in genuine_joint]),
                np.array([coevasion_test.score(attack_sample)] * N_ATTACK),
            ]
        )
        baseline_scores = np.concatenate(
            [
                np.array([conflict_score(d, np.zeros(DIM)) for d in genuine_deltas]),
                np.array([conflict_score(attack_sample, np.zeros(DIM))] * N_ATTACK),
            ]
        )
        labels = np.concatenate([np.zeros(N_GENUINE), np.ones(N_ATTACK)])

        tiers[tier_name] = {
            "drift": (drift_scores, labels),
            "coevasion": (coevasion_scores, labels),
            "baseline": (baseline_scores, labels),
        }

    return tiers


def build_eval_matrix(seed: int = 0) -> dict:
    tiers = _build_scores(seed)

    methods = {
        "snapshot_baseline": lambda scores: scores,
        "drift_only": lambda scores: scores,
        "coevasion_only": lambda scores: scores,
        "full_system": lambda scores: scores,
    }

    score_key_by_method = {
        "snapshot_baseline": "baseline",
        "drift_only": "drift",
        "coevasion_only": "coevasion",
        "full_system": "coevasion",
    }

    result: dict = {}
    for method_name, method_fn in methods.items():
        score_key = score_key_by_method[method_name]
        attack_tiers = {
            tier_name: tier_scores[score_key] for tier_name, tier_scores in tiers.items()
        }
        result[method_name] = run_eval_matrix({method_name: method_fn}, attack_tiers)[method_name]

    return result


if __name__ == "__main__":
    import json

    print(json.dumps(build_eval_matrix(), indent=2))
