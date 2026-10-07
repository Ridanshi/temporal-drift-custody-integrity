"""Tests the elevation test against an attacker who knows about it specifically, rather
than an attacker built only to evade the older sum-based z-test. Section 7.10's "creeping"
attacker holds every hop at the edge-level 90th-percentile threshold, which evades
single-edge drift detection but does nothing to avoid the elevation test's 70th-percentile
elevated bar; every one of its hops is elevated by construction, which is exactly why the
elevation test catches it so quickly. An attacker who knows the elevation test's own bar
has a cheaper option: hold every hop just under that bar instead. This guarantees zero
hops ever count as elevated, so the binomial elevation test never fires regardless of
chain length, at the cost of a lower per-hop ceiling to work with for the z-test, which
still accumulates over the chain.
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
        # attacker stays strictly under the elevated bar; use a small margin below it
        margin = abs(elevated_bar) * 0.01 + 0.01
        attacker_ceiling = elevated_bar - margin

        report[transform_type] = {
            "per_edge_stats": {
                "mean": round(mu, 3), "std": round(sigma, 3),
                "threshold_90th": round(tau, 3), "elevated_bar_70th": round(elevated_bar, 3),
                "attacker_ceiling": round(attacker_ceiling, 3),
            },
            "sweep": {},
        }

        # analytic crossover for this attacker against the z-test alone
        ratio = (attacker_ceiling - mu) / sigma
        analytic_n = (Z_THRESHOLD / ratio) ** 2 if ratio > 0 else None

        for n_hops in chain_lengths:
            genuine_chains = np.array(
                [rng.choice(scores, size=n_hops, replace=True) for _ in range(n_bootstrap)]
            )
            genuine_sums = genuine_chains.sum(axis=1)
            path_mean, path_std = float(genuine_sums.mean()), float(genuine_sums.std(ddof=1))

            fp_count = 0
            for chain, chain_sum in zip(genuine_chains, genuine_sums):
                z = float(path_break_zscore(np.array([chain_sum]), path_mean, path_std)[0])
                n_elev = int(np.sum(chain > elevated_bar))
                p_elev = path_elevation_pvalue(n_hops, n_elev, ELEVATED_PCTILE)
                v = path_combined_verdict(path_break=abs(z) > Z_THRESHOLD, path_elevated=p_elev < ALPHA)
                if v.flagged:
                    fp_count += 1
            combined_fp_rate = fp_count / n_bootstrap

            attack_chain = np.full(n_hops, attacker_ceiling)
            attack_sum = float(attack_chain.sum())
            z_attack = float(path_break_zscore(np.array([attack_sum]), path_mean, path_std)[0])
            n_elev_attack = int(np.sum(attack_chain > elevated_bar))
            p_elev_attack = path_elevation_pvalue(n_hops, n_elev_attack, ELEVATED_PCTILE)
            v_attack = path_combined_verdict(
                path_break=abs(z_attack) > Z_THRESHOLD, path_elevated=p_elev_attack < ALPHA
            )

            report[transform_type]["sweep"][str(n_hops)] = {
                "attack_z": round(z_attack, 3),
                "attack_n_elevated": n_elev_attack,
                "z_flags": bool(abs(z_attack) > Z_THRESHOLD),
                "elevation_flags": bool(p_elev_attack < ALPHA),
                "combined_flagged": v_attack.flagged,
                "genuine_fp_rate": round(combined_fp_rate, 4),
            }

        report[transform_type]["analytic_crossover_hops"] = round(float(analytic_n), 1) if analytic_n else None

    return report


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\DF paper\data\vision"
    print(json.dumps(run_experiment(root), indent=2))
