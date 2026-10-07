"""Tests the two-sided elevation test (custody.propagate.flag_path_elevation_two_sided)
against the elevation-test-aware attacker from Section 7.13, who holds every hop just
under the elevated bar specifically to guarantee zero elevated hops and evade the
one-sided elevation test by construction. The two-sided test adds a low-tail check
(path_elevation_deficit_pvalue): under a genuine chain each hop independently has
probability (1 - elevated_percentile) of landing above the bar, so seeing zero (or very
few) elevated hops out of many is itself improbable under that null. Each tail is tested
at alpha/2 (Bonferroni split) so the overall false-positive rate under the null stays at
the same alpha as the one-sided tests used throughout Sections 7.10-7.13.

Also reconfirms the original maximal creeping attacker (Section 7.10) is still caught
by the two-sided test's high tail, and reports the two-sided test's genuine
false-positive rate against the one-sided test's, to show the Bonferroni split does not
silently inflate false alarms.
"""
import json

import numpy as np

from custody.drift import DriftModel
from custody.propagate import (
    flag_path_elevation,
    flag_path_elevation_two_sided,
    path_break_zscore,
)
from custody.vision_loader import extract_genuine_deltas

TRANSFORM_TYPES = ("whatsapp", "youtube")
ELEVATED_PCTILE = 0.70
Z_THRESHOLD = 3.0
ALPHA = 0.05


def run_experiment(vision_root: str, n_bootstrap: int = 5000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    deltas_by_type = {t: extract_genuine_deltas(vision_root, t) for t in TRANSFORM_TYPES}

    drift_model = DriftModel(n_components=1)
    drift_model.fit(deltas_by_type, percentile=90.0)

    report = {}
    chain_lengths = [1, 3, 5, 7, 9, 11, 15, 20, 30, 50, 75, 100]

    for transform_type in TRANSFORM_TYPES:
        scores = np.array([drift_model.score_edge(transform_type, d) for d in deltas_by_type[transform_type]])
        tau = float(drift_model.thresholds[transform_type])
        mu, sigma = float(scores.mean()), float(scores.std(ddof=1))
        elevated_bar = float(np.percentile(scores, ELEVATED_PCTILE * 100))
        margin = abs(elevated_bar) * 0.01 + 0.01
        attacker_ceiling = elevated_bar - margin

        report[transform_type] = {
            "per_edge_stats": {
                "mean": round(mu, 3), "std": round(sigma, 3),
                "elevated_bar_70th": round(elevated_bar, 3),
                "attacker_ceiling": round(attacker_ceiling, 3),
            },
            "adaptive_attacker_sweep": {},
            "creeping_attacker_sweep": {},
            "genuine_fp": {},
        }

        for n_hops in chain_lengths:
            genuine_chains = np.array(
                [rng.choice(scores, size=n_hops, replace=True) for _ in range(n_bootstrap)]
            )
            genuine_sums = genuine_chains.sum(axis=1)
            path_mean, path_std = float(genuine_sums.mean()), float(genuine_sums.std(ddof=1))

            # --- genuine false-positive rate: one-sided elevation test vs two-sided ---
            one_sided_fp = 0
            two_sided_fp = 0
            for chain in genuine_chains:
                n_elev = int(np.sum(chain > elevated_bar))
                if flag_path_elevation(n_hops, n_elev, ELEVATED_PCTILE, ALPHA):
                    one_sided_fp += 1
                if flag_path_elevation_two_sided(n_hops, n_elev, ELEVATED_PCTILE, ALPHA):
                    two_sided_fp += 1
            report[transform_type]["genuine_fp"][str(n_hops)] = {
                "one_sided_rate": round(one_sided_fp / n_bootstrap, 4),
                "two_sided_rate": round(two_sided_fp / n_bootstrap, 4),
            }

            # --- adaptive attacker: every hop held just under the elevated bar ---
            attack_chain = np.full(n_hops, attacker_ceiling)
            attack_sum = float(attack_chain.sum())
            z_attack = float(path_break_zscore(np.array([attack_sum]), path_mean, path_std)[0])
            n_elev_attack = int(np.sum(attack_chain > elevated_bar))
            one_sided_flag = flag_path_elevation(n_hops, n_elev_attack, ELEVATED_PCTILE, ALPHA)
            two_sided_flag = flag_path_elevation_two_sided(n_hops, n_elev_attack, ELEVATED_PCTILE, ALPHA)
            z_flag = abs(z_attack) > Z_THRESHOLD
            report[transform_type]["adaptive_attacker_sweep"][str(n_hops)] = {
                "n_elevated": n_elev_attack,
                "z": round(z_attack, 3),
                "z_flags": bool(z_flag),
                "one_sided_elevation_flags": bool(one_sided_flag),
                "two_sided_elevation_flags": bool(two_sided_flag),
                "old_combined_flagged": bool(z_flag or one_sided_flag),
                "new_combined_flagged": bool(z_flag or two_sided_flag),
            }

            # --- creeping attacker (Section 7.10/7.11): every hop at its own threshold ---
            creep_chain = np.full(n_hops, tau)
            creep_sum = float(creep_chain.sum())
            z_creep = float(path_break_zscore(np.array([creep_sum]), path_mean, path_std)[0])
            n_elev_creep = int(np.sum(creep_chain > elevated_bar))
            report[transform_type]["creeping_attacker_sweep"][str(n_hops)] = {
                "n_elevated": n_elev_creep,
                "z": round(z_creep, 3),
                "one_sided_elevation_flags": bool(
                    flag_path_elevation(n_hops, n_elev_creep, ELEVATED_PCTILE, ALPHA)
                ),
                "two_sided_elevation_flags": bool(
                    flag_path_elevation_two_sided(n_hops, n_elev_creep, ELEVATED_PCTILE, ALPHA)
                ),
            }

    return report


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\DF paper\data\vision"
    print(json.dumps(run_experiment(root), indent=2))
