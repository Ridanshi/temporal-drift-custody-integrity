import numpy as np
from custody.propagate import (
    cumulative_path_scores,
    elevated_hop_count,
    flag_path_breaks,
    flag_path_elevation,
    flag_path_elevation_deficit,
    flag_path_elevation_two_sided,
    path_elevation_deficit_pvalue,
    path_elevation_pvalue,
)


def test_cumulative_path_scores():
    result = cumulative_path_scores([1.0, 2.0, 3.0])
    assert np.allclose(result, [1.0, 3.0, 6.0])


def test_flag_path_breaks_no_break_when_within_baseline():
    cumulative = np.array([1.0, 2.0, 3.0])
    baseline_mean = np.array([1.0, 2.0, 3.0])
    baseline_std = np.array([0.5, 0.5, 0.5])

    breaks = flag_path_breaks(cumulative, baseline_mean, baseline_std, z_threshold=3.0)

    assert breaks == []


def test_flag_path_breaks_detects_outlier_step():
    cumulative = np.array([1.0, 2.0, 20.0])
    baseline_mean = np.array([1.0, 2.0, 3.0])
    baseline_std = np.array([0.5, 0.5, 0.5])

    breaks = flag_path_breaks(cumulative, baseline_mean, baseline_std, z_threshold=3.0)

    assert breaks == [2]


def test_flag_path_breaks_handles_zero_std():
    cumulative = np.array([5.0])
    baseline_mean = np.array([1.0])
    baseline_std = np.array([0.0])

    breaks = flag_path_breaks(cumulative, baseline_mean, baseline_std, z_threshold=3.0)

    assert breaks == [0]


def test_elevated_hop_count_counts_ranks_above_bar():
    ranks = np.array([0.5, 0.75, 0.95, 0.6, 0.99])

    assert elevated_hop_count(ranks, elevated_percentile=0.70) == 3


def test_elevated_hop_count_zero_when_none_elevated():
    ranks = np.array([0.1, 0.2, 0.3])

    assert elevated_hop_count(ranks, elevated_percentile=0.70) == 0


def test_path_elevation_pvalue_matches_binomial_tail():
    # Under the null, P(all 5 hops elevated | p_null=0.3) = 0.3**5.
    p = path_elevation_pvalue(n_hops=5, n_elevated=5, elevated_percentile=0.70)

    assert np.isclose(p, 0.3**5, atol=1e-6)


def test_path_elevation_pvalue_high_when_few_elevated():
    p = path_elevation_pvalue(n_hops=10, n_elevated=1, elevated_percentile=0.70)

    assert p > 0.5


def test_flag_path_elevation_flags_all_hops_elevated_at_moderate_n():
    # 4 of 4 hops elevated past the 70th percentile: p = 0.3**4 = 0.0081 < 0.05.
    assert flag_path_elevation(n_hops=4, n_elevated=4, elevated_percentile=0.70) is True


def test_flag_path_elevation_does_not_flag_typical_genuine_chain():
    # 1 of 5 hops elevated is close to the expected count under the null (5*0.3=1.5).
    assert flag_path_elevation(n_hops=5, n_elevated=1, elevated_percentile=0.70) is False


def test_path_elevation_deficit_pvalue_matches_binomial_tail():
    # Under the null, P(0 of 9 hops elevated | p_null=0.3) = 0.7**9.
    p = path_elevation_deficit_pvalue(n_hops=9, n_elevated=0, elevated_percentile=0.70)

    assert np.isclose(p, 0.7**9, atol=1e-6)


def test_path_elevation_deficit_pvalue_high_for_typical_genuine_chain():
    p = path_elevation_deficit_pvalue(n_hops=10, n_elevated=3, elevated_percentile=0.70)

    assert p > 0.5


def test_flag_path_elevation_deficit_flags_zero_elevated_at_moderate_n():
    # 0.7**9 = 0.0404 < 0.05, but 0.7**5 = 0.168 is not, so N=9 is the crossover.
    assert flag_path_elevation_deficit(n_hops=9, n_elevated=0, elevated_percentile=0.70) is True
    assert flag_path_elevation_deficit(n_hops=5, n_elevated=0, elevated_percentile=0.70) is False


def test_flag_path_elevation_deficit_does_not_flag_typical_genuine_chain():
    assert flag_path_elevation_deficit(n_hops=10, n_elevated=3, elevated_percentile=0.70) is False


def test_flag_path_elevation_two_sided_flags_high_tail():
    # 4 of 4 hops elevated: same creeping-attacker case flag_path_elevation catches.
    assert flag_path_elevation_two_sided(n_hops=4, n_elevated=4, elevated_percentile=0.70) is True


def test_flag_path_elevation_two_sided_flags_low_tail():
    # 0 of 11 hops elevated: 0.7**11 = 0.0198 < alpha/2 = 0.025 (Bonferroni split), the
    # bar-avoiding attacker flag_path_elevation alone misses regardless of chain length.
    assert flag_path_elevation_two_sided(n_hops=11, n_elevated=0, elevated_percentile=0.70) is True


def test_flag_path_elevation_two_sided_does_not_flag_typical_genuine_chain():
    assert flag_path_elevation_two_sided(n_hops=10, n_elevated=3, elevated_percentile=0.70) is False
