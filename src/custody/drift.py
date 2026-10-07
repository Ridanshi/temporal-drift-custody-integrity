import numpy as np
from sklearn.mixture import GaussianMixture


class DriftModel:
    def __init__(self, n_components: int = 1):
        self.n_components = n_components
        self.models: dict[str, GaussianMixture] = {}
        self.thresholds: dict[str, float] = {}

    def fit(self, deltas_by_type: dict[str, np.ndarray], percentile: float = 95.0) -> None:
        for transform_type, deltas in deltas_by_type.items():
            gmm = GaussianMixture(n_components=self.n_components, random_state=0)
            gmm.fit(deltas)
            self.models[transform_type] = gmm
            scores = -gmm.score_samples(deltas)
            self.thresholds[transform_type] = float(np.percentile(scores, percentile))

    def score_edge(self, transform_type: str, delta: np.ndarray) -> float:
        gmm = self.models[transform_type]
        return float(-gmm.score_samples(delta.reshape(1, -1))[0])

    def is_anomalous(self, transform_type: str, delta: np.ndarray) -> bool:
        return self.score_edge(transform_type, delta) > self.thresholds[transform_type]
