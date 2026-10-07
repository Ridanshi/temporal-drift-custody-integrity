"""Tests a new path-anomaly signal, elevated-hop counting, against the same creeping
attacker used in Section 7.10, to see whether it closes the gap the sum-based z-test
leaves at short chain lengths. Where the z-test asks "is the accumulated deviation
unusual", this test asks "are too many hops individually elevated relative to their own
type's genuine distribution at once", a signal targeted directly at an attacker who keeps
every hop consistently near its own threshold rather than accumulating one large jump.

For a genuine chain, each hop independently has probability (1 - elevated_percentile) of
exceeding the elevated bar by construction, so the count of elevated hops is
Binomial(N, 1 - elevated_percentile) under the null. A creeping attacker whose every hop
sits at its own 90th-percentile threshold (above any elevated bar below the 90th) always
has all N hops elevated, giving a one-sided binomial p-value of elevated_percentile^N,
which can clear significance at much smaller N than the sum-based z-test's sqrt(N) scaling
requires.

The genuine chain bootstrap draw is shared across every elevated-percentile setting (drawn
once per chain length, before the percentile loop) so the z-test's own crossover point,
which does not depend on elevated_percentile at all, is reported consistently rather than
re-derived with fresh resampling noise each time.
"""
import json

import numpy as np

from custody.drift import DriftModel
from custody.propagate import path_break_zscore, path_elevation_pvalue
from custody.vision_loader import extract_genuine_deltas

TRANSFORM_TYPES = ("whatsapp", "youtube")


def run_experiment(vision_root: str, n_bootstrap: int = 5000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    deltas_by_type = {t: extract_genuine_deltas(vision_root, t) for t in TRANSFORM_TYPES}

    drift_model = DriftModel(n_components=1)
    drift_model.fit(deltas_by_type, percentile=90.0)

    report = {}
    chain_lengths = [1, 2, 3, 4, 5, 6, 7, 9, 11, 15, 20]
    elevated_percentiles = [0.60, 0.70, 0.80]

    for transform_type in TRANSFORM_TYPES:
        scores = np.array([drift_model.score_edge(transform_type, d) for d in deltas_by_type[transform_type]])
        tau = float(drift_model.thresholds[transform_type])
        mu, sigma = float(scores.mean()), float(scores.std(ddof=1))

        report[transform_type] = {
            "per_edge_stats": {"mean": round(mu, 3), "std": round(sigma, 3), "threshold": round(tau, 3)},
            "z_test": {},
        }
        elev_reports = {f"elevated_pctile_{p}": {} for p in elevated_percentiles}

        for n_hops in chain_lengths:
            # Genuine chains drawn once per chain length, shared across every
            # elevated-percentile setting below.
            genuine_chains = np.array(
                [rng.choice(scores, size=n_hops, replace=True) for _ in range(n_bootstrap)]
            )
            genuine_sums = genuine_chains.sum(axis=1)
            path_mean, path_std = float(genuine_sums.mean()), float(genuine_sums.std(ddof=1))

            genuine_z = path_break_zscore(genuine_sums, path_mean, path_std)
            z_fp_rate = float((np.abs(genuine_z) > 3.0).mean())

            attack_sum = n_hops * tau
            attack_z = float(path_break_zscore(np.array([attack_sum]), path_mean, path_std)[0])

            report[transform_type]["z_test"][str(n_hops)] = {
                "attack_z": round(attack_z, 3),
                "attack_flagged": bool(abs(attack_z) > 3.0),
                "genuine_fp_rate": round(z_fp_rate, 4),
            }

            for elev_pct in elevated_percentiles:
                elevated_bar = float(np.percentile(scores, elev_pct * 100))

                genuine_elevated_counts = (genuine_chains > elevated_bar).sum(axis=1)
                genuine_elev_pvals = np.array(
                    [path_elevation_pvalue(n_hops, int(k), elev_pct) for k in genuine_elevated_counts]
                )
                elev_fp_rate = float((genuine_elev_pvals < 0.05).mean())

                attack_elevated_count = n_hops if tau > elevated_bar else 0
                attack_elev_pval = path_elevation_pvalue(n_hops, attack_elevated_count, elev_pct)

                elev_reports[f"elevated_pctile_{elev_pct}"][str(n_hops)] = {
                    "attack_elevated_count": attack_elevated_count,
                    "attack_pvalue": round(float(attack_elev_pval), 5),
                    "attack_flagged": bool(attack_elev_pval < 0.05),
                    "genuine_fp_rate": round(elev_fp_rate, 4),
                }

        report[transform_type].update(elev_reports)

    return report


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\DF paper\data\vision"
    print(json.dumps(run_experiment(root), indent=2))
