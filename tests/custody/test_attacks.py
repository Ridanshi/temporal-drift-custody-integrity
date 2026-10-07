import numpy as np
from custody.attacks import naive_tamper, heuristic_coevasion, adaptive_coevasion
from custody.coevasion import CoEvasionTest
from custody.synthetic import generate_joint_samples


def test_naive_tamper_shifts_away_from_genuine():
    genuine = np.array([1.0, 1.0])
    tampered = naive_tamper(genuine, magnitude=5.0, seed=0)
    assert not np.allclose(tampered, genuine)


def test_naive_tamper_deterministic_with_seed():
    genuine = np.array([1.0, 1.0])
    t1 = naive_tamper(genuine, magnitude=5.0, seed=1)
    t2 = naive_tamper(genuine, magnitude=5.0, seed=1)
    assert np.allclose(t1, t2)


def test_heuristic_coevasion_stays_close_to_genuine_mean():
    genuine_mean = np.array([0.0, 0.0])
    forged = heuristic_coevasion(genuine_mean, noise=0.3, seed=0)
    assert np.linalg.norm(forged - genuine_mean) < 3.0


def test_adaptive_coevasion_targets_undetectable_middle_distance():
    samples = generate_joint_samples(n=500, dim=2, seed=0)
    test = CoEvasionTest()
    test.fit(samples, low_percentile=1.0, high_percentile=99.0)
    target_mean = test.mean

    forged = adaptive_coevasion(test, target_mean, n_steps=100, seed=0)
    forged_score = test.score(forged)

    assert test.tau_low < forged_score < test.tau_high


def test_adaptive_coevasion_deterministic_with_seed():
    samples = generate_joint_samples(n=500, dim=2, seed=0)
    test = CoEvasionTest()
    test.fit(samples)

    f1 = adaptive_coevasion(test, test.mean, n_steps=50, seed=7)
    f2 = adaptive_coevasion(test, test.mean, n_steps=50, seed=7)

    assert np.allclose(f1, f2)
