import numpy as np
from custody.calibration import expected_calibration_error, scores_to_pseudo_probabilities


def test_perfectly_calibrated_gives_zero_ece():
    probs = np.array([0.1, 0.1, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    ece = expected_calibration_error(probs, labels, n_bins=5)
    assert ece < 0.15


def test_badly_calibrated_gives_high_ece():
    probs = np.array([0.9, 0.9, 0.9, 0.9])
    labels = np.array([0, 0, 0, 0])
    ece = expected_calibration_error(probs, labels, n_bins=5)
    assert ece > 0.7


def test_scores_to_pseudo_probabilities_bounded():
    scores = np.array([0.0, 5.0, 10.0, 2.5])
    probs = scores_to_pseudo_probabilities(scores)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    assert np.isclose(probs.min(), 0.0)
    assert np.isclose(probs.max(), 1.0)


def test_scores_to_pseudo_probabilities_constant_input():
    scores = np.array([3.0, 3.0, 3.0])
    probs = scores_to_pseudo_probabilities(scores)
    assert np.all(probs == 0.5)
