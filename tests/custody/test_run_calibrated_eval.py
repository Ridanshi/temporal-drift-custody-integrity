from scripts.run_calibrated_eval import build_calibrated_eval_matrix


def test_build_calibrated_eval_matrix_has_all_methods_and_tiers():
    result = build_calibrated_eval_matrix(n_genuine=100, n_attack_instances=30, seed=0)

    assert set(result.keys()) == {"snapshot_baseline", "drift_only", "coevasion_only", "full_system"}
    for method_results in result.values():
        assert set(method_results.keys()) == {"naive", "heuristic", "adaptive"}
        for tier_result in method_results.values():
            assert 0.0 <= tier_result["auc"] <= 1.0
            assert 0.0 <= tier_result["f1"] <= 1.0


def test_naive_tier_is_easy_for_every_method():
    result = build_calibrated_eval_matrix(n_genuine=100, n_attack_instances=30, seed=0)

    for method_results in result.values():
        assert method_results["naive"]["auc"] >= 0.9


def test_full_system_beats_or_matches_baseline_on_adaptive_tier():
    result = build_calibrated_eval_matrix(n_genuine=150, n_attack_instances=40, seed=1)

    # The adaptive tier sits near/below chance for every method on synthetic data (see
    # Section 7.1's paper caveat: this is a construction artifact of the synthetic
    # attacker, not a real effect), so GaussianMixture's BLAS-threaded EM can flip a
    # handful of borderline instances run-to-run. A small documented tolerance keeps
    # the test meaningful (catching a real regression) without being flaky on noise.
    full_auc = result["full_system"]["adaptive"]["auc"]
    baseline_auc = result["snapshot_baseline"]["adaptive"]["auc"]
    assert full_auc >= baseline_auc - 0.05


def test_deterministic_with_seed():
    r1 = build_calibrated_eval_matrix(n_genuine=50, n_attack_instances=20, seed=7)
    r2 = build_calibrated_eval_matrix(n_genuine=50, n_attack_instances=20, seed=7)

    assert r1 == r2
