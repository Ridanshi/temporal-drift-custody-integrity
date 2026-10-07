import numpy as np
from custody.stats import mcnemar_test


def test_symmetric_disagreement_gives_zero_chi2():
    correct_a = np.array([True, False, True, False])
    correct_b = np.array([False, True, False, True])
    result = mcnemar_test(correct_a, correct_b)
    assert result["b"] == 2
    assert result["c"] == 2
    assert result["chi2"] < 1.0
    assert result["p_value"] > 0.5


def test_one_sided_disagreement_is_significant():
    correct_a = np.array([True] * 20 + [False] * 5)
    correct_b = np.array([False] * 20 + [False] * 5)
    result = mcnemar_test(correct_a, correct_b)
    assert result["b"] == 20
    assert result["c"] == 0
    assert result["chi2"] > 0
    assert result["p_value"] < 0.05


def test_identical_predictions_give_zero_disagreement():
    correct_a = np.array([True, True, False, False])
    correct_b = np.array([True, True, False, False])
    result = mcnemar_test(correct_a, correct_b)
    assert result["b"] == 0
    assert result["c"] == 0
    assert result["chi2"] == 0.0
    assert result["p_value"] == 1.0
