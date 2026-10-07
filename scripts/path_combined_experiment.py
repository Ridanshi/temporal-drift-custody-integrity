"""Integrates the elevated-hop-count test (Section 7.11) with the sum-based z-test
(Section 7.10) into a single combined detector (custody.pipeline.path_combined_verdict,
flagged if either signal fires) and tests it against two different attacker patterns:

1. The same maximal "creeping" attacker from Sections 7.10/7.11 (every hop at its own
   threshold), to confirm the combined detector inherits the elevation test's short-chain
   power rather than being diluted by including the weaker z-test.
2. A new "single elevated hop" attacker: exactly one hop set to a moderately elevated
   percentile of its own genuine distribution (below its own edge-level 90th-percentile
   drift threshold, so single-edge drift-only still misses it, and below what the
   elevation test's binomial count treats as improbable for a single hop), the rest of the
   chain genuine. This tests the z-test's claimed comparative advantage for a one-time
   jump concentrated on a single hop directly, rather than only asserting it.
"""
import json

import numpy as np

from custody.drift import DriftModel
from custody.pipeline import path_combined_verdict
from custody.propagate import path_break_zscore, path_elevation_pvalue
from custody.vision_loader import extract_genuine_deltas

TRANSFORM_TYPES = ("whatsapp", "youtube")
ELEVATED_PCTILE = 0.70
Z_THRESHOLD = 3.0
ALPHA = 0.05


def verdict_for_chain(chain_sum, chain, path_mean, path_std, elevated_bar, n_hops):
    z = float(path_break_zscore(np.array([chain_sum]), path_mean, path_std)[0])
    path_break = abs(z) > Z_THRESHOLD
    n_elevated = int(np.sum(chain > elevated_bar))
    p_elev = path_elevation_pvalue(n_hops, n_elevated, ELEVATED_PCTILE)
    path_elevated = p_elev < ALPHA
    v = path_combined_verdict(path_break=path_break, path_elevated=path_elevated)
    return v, z, n_elevated, p_elev


def run_experiment(vision_root: str, n_bootstrap: int = 3000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    deltas_by_type = {t: extract_genuine_deltas(vision_root, t) for t in TRANSFORM_TYPES}

    drift_model = DriftModel(n_components=1)
    drift_model.fit(deltas_by_type, percentile=90.0)

    report = {}
    chain_lengths = [3, 5, 7, 9, 15, 20]
    single_hop_percentiles = [0.75, 0.80, 0.85]

    for transform_type in TRANSFORM_TYPES:
        scores = np.array([drift_model.score_edge(transform_type, d) for d in deltas_by_type[transform_type]])
        tau = float(drift_model.thresholds[transform_type])
        elevated_bar = float(np.percentile(scores, ELEVATED_PCTILE * 100))

        report[transform_type] = {"creeping_attacker": {}, "single_hop_attacker": {}}

        for n_hops in chain_lengths:
            genuine_chains = np.array(
                [rng.choice(scores, size=n_hops, replace=True) for _ in range(n_bootstrap)]
            )
            genuine_sums = genuine_chains.sum(axis=1)
            path_mean, path_std = float(genuine_sums.mean()), float(genuine_sums.std(ddof=1))

            # --- genuine false-positive rate of the COMBINED detector at this N ---
            fp_count = 0
            for chain, chain_sum in zip(genuine_chains, genuine_sums):
                v, _, _, _ = verdict_for_chain(chain_sum, chain, path_mean, path_std, elevated_bar, n_hops)
                if v.flagged:
                    fp_count += 1
            combined_fp_rate = fp_count / n_bootstrap

            # --- creeping attacker: every hop at its own threshold ---
            creep_chain = np.full(n_hops, tau)
            creep_sum = float(creep_chain.sum())
            v_creep, z_creep, n_elev_creep, p_creep = verdict_for_chain(
                creep_sum, creep_chain, path_mean, path_std, elevated_bar, n_hops
            )
            report[transform_type]["creeping_attacker"][str(n_hops)] = {
                "z": round(z_creep, 3),
                "n_elevated": n_elev_creep,
                "elevation_pvalue": round(p_creep, 5),
                "combined_flagged": v_creep.flagged,
                "reasons": v_creep.reasons,
                "genuine_fp_rate": round(combined_fp_rate, 4),
            }

            # --- single-hop attacker: one hop pinned to a spike value, the other N-1
            # hops bootstrap-resampled from the real genuine pool (not pinned to a fixed
            # "typical" value), since these scores are heavily right-skewed and any fixed
            # choice (mean or median) misrepresents genuine chains' true sampling
            # variability, silently penalizing or inflating the constructed attack.
            # n_attack_draws independent attack chains are built per spike percentile so
            # we report the fraction flagged, not a single noisy instance.
            n_attack_draws = 1000
            single_hop_results = {}
            for hop_pctile in single_hop_percentiles:
                hop_value = float(np.percentile(scores, hop_pctile * 100))
                rest_hops = rng.choice(scores, size=(n_attack_draws, n_hops - 1), replace=True)
                attack_chains = np.concatenate(
                    [np.full((n_attack_draws, 1), hop_value), rest_hops], axis=1
                )
                attack_sums = attack_chains.sum(axis=1)

                z_arr = path_break_zscore(attack_sums, path_mean, path_std)
                z_flag_rate = float((np.abs(z_arr) > Z_THRESHOLD).mean())

                n_elev_arr = (attack_chains > elevated_bar).sum(axis=1)
                elev_pvals = np.array(
                    [path_elevation_pvalue(n_hops, int(k), ELEVATED_PCTILE) for k in n_elev_arr]
                )
                elev_flag_rate = float((elev_pvals < ALPHA).mean())

                combined_flag_rate = float(
                    ((np.abs(z_arr) > Z_THRESHOLD) | (elev_pvals < ALPHA)).mean()
                )

                single_hop_results[str(hop_pctile)] = {
                    "hop_value": round(hop_value, 3),
                    "below_own_threshold": hop_value < tau,
                    "mean_n_elevated": round(float(n_elev_arr.mean()), 2),
                    "z_test_detection_rate": round(z_flag_rate, 4),
                    "elevation_test_detection_rate": round(elev_flag_rate, 4),
                    "combined_detection_rate": round(combined_flag_rate, 4),
                }
            report[transform_type]["single_hop_attacker"][str(n_hops)] = single_hop_results

    return report


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\DF paper\data\vision"
    print(json.dumps(run_experiment(root), indent=2))
