import zlib

import numpy as np


def _type_mean(transform_type: str, dim: int) -> np.ndarray:
    type_seed = zlib.crc32(transform_type.encode()) % (2**32)
    rng = np.random.default_rng(type_seed)
    return rng.normal(0, 1, size=dim)


def generate_genuine_deltas(transform_type: str, n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = _type_mean(transform_type, dim)
    cov = np.eye(dim) * 0.1
    return rng.multivariate_normal(mean, cov, size=n)


def generate_joint_samples(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = np.zeros(dim)
    cov = np.eye(dim) * 0.05
    return rng.multivariate_normal(mean, cov, size=n)
