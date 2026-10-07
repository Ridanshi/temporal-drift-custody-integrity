import numpy as np
from custody.coevasion import CoEvasionTest


def _joint_samples(n=500, seed=0):
    rng = np.random.default_rng(seed)
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    return rng.multivariate_normal(mean, cov, size=n)


def test_typical_genuine_sample_not_flagged():
    test = CoEvasionTest()
    samples = _joint_samples(seed=1)
    test.fit(samples, low_percentile=1.0, high_percentile=99.0)

    typical = samples[0]

    assert test.flag(typical) is None


def test_far_outlier_flagged_as_mismatch():
    test = CoEvasionTest()
    samples = _joint_samples(seed=1)
    test.fit(samples, low_percentile=1.0, high_percentile=99.0)

    outlier = np.array([50.0, -50.0])

    assert test.flag(outlier) == "mismatch"


def test_too_perfect_sample_flagged():
    test = CoEvasionTest()
    samples = _joint_samples(seed=1)
    test.fit(samples, low_percentile=1.0, high_percentile=99.0)

    exact_center = test.mean

    assert test.flag(exact_center) == "too_perfect"


def test_score_is_finite_float():
    test = CoEvasionTest()
    samples = _joint_samples(seed=1)
    test.fit(samples)

    score = test.score(samples[0])

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_anomaly_score_zero_inside_normal_band():
    test = CoEvasionTest()
    samples = _joint_samples(seed=1)
    test.fit(samples, low_percentile=1.0, high_percentile=99.0)

    typical = samples[0]

    assert test.anomaly_score(typical) == 0.0


def test_anomaly_score_positive_for_mismatch_outlier():
    test = CoEvasionTest()
    samples = _joint_samples(seed=1)
    test.fit(samples, low_percentile=1.0, high_percentile=99.0)

    outlier = np.array([50.0, -50.0])

    assert test.anomaly_score(outlier) > 0.0


def test_anomaly_score_positive_for_too_perfect_sample():
    test = CoEvasionTest()
    samples = _joint_samples(seed=1)
    test.fit(samples, low_percentile=1.0, high_percentile=99.0)

    exact_center = test.mean

    assert test.anomaly_score(exact_center) > 0.0
