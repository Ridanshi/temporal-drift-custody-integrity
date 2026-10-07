import numpy as np
from custody.ml_baseline import IsolationForestBaseline, OneClassSVMBaseline


def _genuine_cluster(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=[0, 0, 0, 0], cov=np.eye(4) * 0.05, size=n)


def test_isolation_forest_scores_outlier_higher_than_genuine():
    model = IsolationForestBaseline(random_state=0)
    genuine = _genuine_cluster(seed=1)
    model.fit(genuine)

    typical_score = model.anomaly_score(genuine[0])
    outlier_score = model.anomaly_score(np.array([10.0, -10.0, 10.0, -10.0]))

    assert outlier_score > typical_score


def test_one_class_svm_scores_outlier_higher_than_genuine():
    model = OneClassSVMBaseline(nu=0.1)
    genuine = _genuine_cluster(seed=2)
    model.fit(genuine)

    typical_score = model.anomaly_score(genuine[0])
    outlier_score = model.anomaly_score(np.array([10.0, -10.0, 10.0, -10.0]))

    assert outlier_score > typical_score


def test_isolation_forest_score_is_finite_float():
    model = IsolationForestBaseline(random_state=0)
    genuine = _genuine_cluster(seed=3)
    model.fit(genuine)

    score = model.anomaly_score(genuine[0])

    assert isinstance(score, float)
    assert np.isfinite(score)
