"""Tests the multi-hop path-propagation component in isolation, which no evaluation in
the paper exercises so far (VISION triplets are single-hop; Section 7.1's synthetic
construction is also single-edge). Builds N-hop custody chains from real per-edge
drift-score statistics (mean, std, 90th-percentile threshold, fit on real VISION genuine
deltas), then asks: can a "creeping" attacker, who keeps every individual hop just at or
under its own edge-level threshold (so a single-edge/drift-only test never flags any one
hop), still be caught by the cumulative path-break z-test, and at what chain length does
that become possible?

For N i.i.d. hops of one transform type (mean mu, std sigma, threshold tau > mu):
  max creeping-attack path sum   = N * tau
  genuine path sum mean          = N * mu
  genuine path sum std           = sigma * sqrt(N)
  z of the maximal creeping attack = sqrt(N) * (tau - mu) / sigma
which grows as sqrt(N): longer chains make the cumulative test strictly more powerful
against a per-hop-bounded attacker, even though no single hop ever trips its own alarm.
"""
import json

import numpy as np

from custody.drift import DriftModel
from custody.propagate import path_break_zscore
from custody.vision_loader import extract_genuine_deltas

TRANSFORM_TYPES = ("whatsapp", "youtube")


def run_path_propagation_experiment(vision_root: str, n_bootstrap: int = 5000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    deltas_by_type = {t: extract_genuine_deltas(vision_root, t) for t in TRANSFORM_TYPES}

    drift_model = DriftModel(n_components=1)
    drift_model.fit(deltas_by_type, percentile=90.0)

    report = {"per_edge_stats": {}, "chain_length_sweep": {}}

    for transform_type in TRANSFORM_TYPES:
        scores = np.array([drift_model.score_edge(transform_type, d) for d in deltas_by_type[transform_type]])
        mu, sigma = float(scores.mean()), float(scores.std(ddof=1))
        tau = float(drift_model.thresholds[transform_type])
        report["per_edge_stats"][transform_type] = {
            "mean": round(mu, 3), "std": round(sigma, 3), "threshold_90th_pctile": round(tau, 3),
        }

        chain_lengths = [1, 2, 3, 5, 7, 9, 11, 15, 20]
        sweep = {}
        for n_hops in chain_lengths:
            # Genuine chains: bootstrap-resample n_hops real per-edge scores, sum them.
            genuine_sums = np.array(
                [rng.choice(scores, size=n_hops, replace=True).sum() for _ in range(n_bootstrap)]
            )
            path_mean, path_std = float(genuine_sums.mean()), float(genuine_sums.std(ddof=1))

            # Maximal creeping attack: every hop exactly at (not over) its own threshold.
            creeping_sum = n_hops * tau
            z_creeping = float(path_break_zscore(np.array([creeping_sum]), path_mean, path_std)[0])

            # Empirical genuine false-positive rate at z_threshold=3.0, for reference.
            genuine_z = path_break_zscore(genuine_sums, path_mean, path_std)
            fp_rate = float((np.abs(genuine_z) > 3.0).mean())

            sweep[str(n_hops)] = {
                "genuine_path_mean": round(path_mean, 2),
                "genuine_path_std": round(path_std, 2),
                "max_creeping_attack_sum": round(creeping_sum, 2),
                "max_creeping_attack_z": round(z_creeping, 3),
                "creeping_attack_flagged_at_z3": bool(abs(z_creeping) > 3.0),
                "genuine_false_positive_rate_at_z3": round(fp_rate, 4),
            }
        report["chain_length_sweep"][transform_type] = sweep

    # Find the minimum chain length at which the creeping attack becomes flaggable, per type.
    report["crossover_chain_length"] = {}
    for transform_type in TRANSFORM_TYPES:
        mu = report["per_edge_stats"][transform_type]["mean"]
        sigma = report["per_edge_stats"][transform_type]["std"]
        tau = report["per_edge_stats"][transform_type]["threshold_90th_pctile"]
        # analytic: z(N) = sqrt(N) * (tau - mu) / sigma >= 3.0
        ratio = (tau - mu) / sigma
        n_needed = (3.0 / ratio) ** 2 if ratio > 0 else None
        report["crossover_chain_length"][transform_type] = {
            "analytic_min_hops": round(float(n_needed), 1) if n_needed else None,
        }

    return report


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\DF paper\data\vision"
    print(json.dumps(run_path_propagation_experiment(root), indent=2))
