import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class IsolationForestBaseline:
    def __init__(self, random_state: int = 0, contamination="auto"):
        self.model = IsolationForest(random_state=random_state, contamination=contamination)

    def fit(self, genuine_samples: np.ndarray) -> None:
        self.model.fit(genuine_samples)

    def anomaly_score(self, sample: np.ndarray) -> float:
        return float(-self.model.decision_function(sample.reshape(1, -1))[0])


class OneClassSVMBaseline:
    def __init__(self, nu: float = 0.1, kernel: str = "rbf", gamma="scale"):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit(self, genuine_samples: np.ndarray) -> None:
        self.model.fit(genuine_samples)

    def anomaly_score(self, sample: np.ndarray) -> float:
        return float(-self.model.decision_function(sample.reshape(1, -1))[0])
