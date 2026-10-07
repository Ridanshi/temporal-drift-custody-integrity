import numpy as np
from custody.conflict import conflict_score, snapshot_baseline_flag


def test_conflict_score_zero_when_identical():
    declared = np.array([1.0, 2.0, 3.0])
    inferred = np.array([1.0, 2.0, 3.0])

    assert conflict_score(declared, inferred) == 0.0


def test_conflict_score_positive_when_different():
    declared = np.array([1.0, 2.0])
    inferred = np.array([4.0, 6.0])

    assert conflict_score(declared, inferred) == 5.0


def test_snapshot_baseline_flag_true_above_threshold():
    declared = np.array([0.0, 0.0])
    inferred = np.array([10.0, 0.0])

    assert snapshot_baseline_flag(declared, inferred, threshold=5.0) is True


def test_snapshot_baseline_flag_false_below_threshold():
    declared = np.array([0.0, 0.0])
    inferred = np.array([1.0, 0.0])

    assert snapshot_baseline_flag(declared, inferred, threshold=5.0) is False
