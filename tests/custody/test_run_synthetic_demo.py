from scripts.run_synthetic_demo import build_eval_matrix


def test_build_eval_matrix_has_all_methods_and_tiers():
    result = build_eval_matrix(seed=0)

    assert set(result.keys()) == {"snapshot_baseline", "drift_only", "coevasion_only", "full_system"}
    for method_results in result.values():
        assert set(method_results.keys()) == {"naive", "heuristic", "adaptive"}
        for tier_result in method_results.values():
            assert 0.0 <= tier_result["auc"] <= 1.0
            assert 0.0 <= tier_result["f1"] <= 1.0


def test_full_system_beats_baseline_on_adaptive_tier():
    result = build_eval_matrix(seed=0)

    assert result["full_system"]["adaptive"]["auc"] >= result["snapshot_baseline"]["adaptive"]["auc"]
