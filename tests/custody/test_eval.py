import numpy as np
from custody.eval import evaluate, run_eval_matrix


def test_evaluate_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    result = evaluate(scores, labels, threshold=0.5)

    assert result["auc"] == 1.0
    assert result["f1"] == 1.0


def test_evaluate_returns_finite_values_on_random_scores():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=50)
    labels = rng.integers(0, 2, size=50)

    result = evaluate(scores, labels, threshold=0.5)

    assert np.isfinite(result["auc"])
    assert np.isfinite(result["f1"])


def test_run_eval_matrix_shape():
    def method_a(scores):
        return scores

    def method_b(scores):
        return scores * 2

    tiers = {
        "naive": (np.array([0.1, 0.2, 0.9, 0.8]), np.array([0, 0, 1, 1])),
        "adaptive": (np.array([0.4, 0.5, 0.6, 0.7]), np.array([0, 0, 1, 1])),
    }
    methods = {"method_a": method_a, "method_b": method_b}

    result = run_eval_matrix(methods, tiers, threshold=0.5)

    assert set(result.keys()) == {"method_a", "method_b"}
    assert set(result["method_a"].keys()) == {"naive", "adaptive"}
    assert "auc" in result["method_a"]["naive"]
    assert "f1" in result["method_a"]["naive"]
