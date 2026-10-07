import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (scores > threshold).astype(int)
    auc = roc_auc_score(labels, scores)
    f1 = f1_score(labels, preds)
    return {"auc": float(auc), "f1": float(f1)}


def run_eval_matrix(
    methods: dict, attack_tiers: dict, threshold: float = 0.5
) -> dict:
    results: dict = {}
    for method_name, method_fn in methods.items():
        results[method_name] = {}
        for tier_name, (scores, labels) in attack_tiers.items():
            transformed_scores = np.asarray(method_fn(scores), dtype=float)
            results[method_name][tier_name] = evaluate(transformed_scores, labels, threshold)
    return results
