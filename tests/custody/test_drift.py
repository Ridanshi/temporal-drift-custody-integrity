import numpy as np
from custody.drift import DriftModel


def _genuine_deltas(mean, n=200, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.eye(len(mean)) * 0.05
    return rng.multivariate_normal(mean, cov, size=n)


def test_fit_and_score_genuine_delta_is_not_anomalous():
    model = DriftModel(n_components=1)
    deltas_by_type = {"re_encode": _genuine_deltas(mean=[1.0, 1.0], seed=1)}
    model.fit(deltas_by_type, percentile=95.0)

    genuine_sample = _genuine_deltas(mean=[1.0, 1.0], n=1, seed=2)[0]

    assert model.is_anomalous("re_encode", genuine_sample) is False


def test_far_outlier_delta_is_anomalous():
    model = DriftModel(n_components=1)
    deltas_by_type = {"re_encode": _genuine_deltas(mean=[1.0, 1.0], seed=1)}
    model.fit(deltas_by_type, percentile=95.0)

    outlier = np.array([50.0, -50.0])

    assert model.is_anomalous("re_encode", outlier) is True


def test_score_edge_is_finite_float():
    model = DriftModel(n_components=1)
    deltas_by_type = {"upload": _genuine_deltas(mean=[0.0, 0.0], seed=3)}
    model.fit(deltas_by_type)

    score = model.score_edge("upload", np.array([0.0, 0.0]))

    assert isinstance(score, float)
    assert np.isfinite(score)
