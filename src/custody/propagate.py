import numpy as np
from scipy.stats import binomtest


def cumulative_path_scores(edge_scores: list[float]) -> np.ndarray:
    return np.cumsum(np.asarray(edge_scores, dtype=float))


def elevated_hop_count(percentile_ranks: np.ndarray, elevated_percentile: float = 0.70) -> int:
    """Number of hops whose score exceeds the given percentile of their own edge type's
    genuine score distribution. percentile_ranks are each hop's genuine-distribution
    percentile rank in [0, 1], not raw scores."""
    return int(np.sum(np.asarray(percentile_ranks) > elevated_percentile))


def path_elevation_pvalue(
    n_hops: int, n_elevated: int, elevated_percentile: float = 0.70
) -> float:
    """One-sided binomial test: under a genuine chain, each hop independently has
    probability (1 - elevated_percentile) of landing above the elevated bar by
    construction. Tests whether n_elevated-or-more hops out of n_hops is improbable
    under that null, which targets a "creeping" attacker who keeps every hop
    consistently near its own threshold rather than accumulating one large deviation."""
    p_null = 1.0 - elevated_percentile
    result = binomtest(n_elevated, n_hops, p_null, alternative="greater")
    return float(result.pvalue)


def flag_path_elevation(
    n_hops: int, n_elevated: int, elevated_percentile: float = 0.70, alpha: float = 0.05
) -> bool:
    return path_elevation_pvalue(n_hops, n_elevated, elevated_percentile) < alpha


def path_elevation_deficit_pvalue(
    n_hops: int, n_elevated: int, elevated_percentile: float = 0.70
) -> float:
    """One-sided binomial test on the low tail: under a genuine chain, each hop
    independently has probability (1 - elevated_percentile) of landing above the
    elevated bar by construction. Tests whether n_elevated-or-fewer hops out of n_hops
    is improbable under that null, which targets an attacker who deliberately holds
    every hop just under the elevated bar to guarantee zero hops ever register as
    elevated, evading path_elevation_pvalue by construction regardless of chain
    length."""
    p_null = 1.0 - elevated_percentile
    result = binomtest(n_elevated, n_hops, p_null, alternative="less")
    return float(result.pvalue)


def flag_path_elevation_deficit(
    n_hops: int, n_elevated: int, elevated_percentile: float = 0.70, alpha: float = 0.05
) -> bool:
    return path_elevation_deficit_pvalue(n_hops, n_elevated, elevated_percentile) < alpha


def flag_path_elevation_two_sided(
    n_hops: int, n_elevated: int, elevated_percentile: float = 0.70, alpha: float = 0.05
) -> bool:
    """Two-sided elevation test: flags both an unnaturally HIGH elevated-hop count (the
    creeping attacker path_elevation_pvalue targets) and an unnaturally LOW one (an
    attacker who avoids the elevated bar on every hop), mirroring how the co-evasion
    test treats both mismatch and suspiciously-perfect alignment as anomalous. Each
    one-sided tail is tested at alpha/2 (Bonferroni split) so the overall false-positive
    rate under the null stays at alpha."""
    half_alpha = alpha / 2.0
    return (
        path_elevation_pvalue(n_hops, n_elevated, elevated_percentile) < half_alpha
        or path_elevation_deficit_pvalue(n_hops, n_elevated, elevated_percentile) < half_alpha
    )


def path_break_zscore(
    cumulative: np.ndarray, baseline_mean: np.ndarray, baseline_std: np.ndarray
) -> np.ndarray:
    safe_std = np.where(baseline_std == 0, 1e-9, baseline_std)
    return (cumulative - baseline_mean) / safe_std


def flag_path_breaks(
    cumulative: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    z_threshold: float = 3.0,
) -> list[int]:
    z = path_break_zscore(cumulative, baseline_mean, baseline_std)
    return [i for i, val in enumerate(z) if abs(val) > z_threshold]
