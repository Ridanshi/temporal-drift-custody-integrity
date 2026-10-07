import numpy as np
from custody.synthetic import generate_genuine_deltas, generate_joint_samples


def test_generate_genuine_deltas_shape():
    deltas = generate_genuine_deltas("re_encode", n=50, dim=3, seed=0)
    assert deltas.shape == (50, 3)


def test_generate_genuine_deltas_deterministic_with_seed():
    d1 = generate_genuine_deltas("re_encode", n=10, dim=2, seed=42)
    d2 = generate_genuine_deltas("re_encode", n=10, dim=2, seed=42)
    assert np.allclose(d1, d2)


def test_generate_genuine_deltas_differ_by_transform_type():
    d1 = generate_genuine_deltas("re_encode", n=200, dim=2, seed=0)
    d2 = generate_genuine_deltas("upload", n=200, dim=2, seed=0)
    assert not np.allclose(d1.mean(axis=0), d2.mean(axis=0))


def test_generate_joint_samples_shape():
    samples = generate_joint_samples(n=100, dim=4, seed=0)
    assert samples.shape == (100, 4)
