"""Statistical remediation for the leave-one-device-out adaptive-tier comparison
(Section 7.2.4). Two additions the crude mean +/- std does not provide:

1. A paired cluster bootstrap (resampling folds/devices, not instances) confidence
   interval on the mean per-fold AUC difference (full system minus baseline), for the
   adaptive tier on each edge type. Resampling at the fold level respects that each
   fold's four/six values come from one device's train/test split, not four
   independent instances, avoiding the pseudo-replication error a naive instance-level
   bootstrap would make.
2. A power analysis: given the observed effect size and fold-to-fold variance, how many
   held-out folds (real genuine devices) would be needed to detect the observed
   difference at 80% power, using a paired t-test as the reference test.
"""
import json

import numpy as np
from scipy import stats

import sys

default_path = r"D:\DF paper\scripts\held_out_eval_n15_result.json"
with open(sys.argv[1] if len(sys.argv) > 1 else default_path) as f:
    R = json.load(f)

RNG = np.random.default_rng(0)
N_BOOT = 20000


def cluster_bootstrap_ci(diffs, n_boot=N_BOOT, alpha=0.05):
    diffs = np.array(diffs)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = RNG.choice(diffs, size=n, replace=True)
        boot_means[b] = sample.mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(diffs.mean()), float(diffs.std(ddof=1))


def paired_ttest(diffs):
    diffs = np.array(diffs)
    t_stat, p_value = stats.ttest_1samp(diffs, popmean=0.0)
    return float(t_stat), float(p_value)


def n_folds_for_power(mean_diff, sd_diff, power=0.80, alpha=0.05):
    """Sample size (paired t-test, two-sided) needed to detect the observed mean
    difference at the observed SD, using the standard noncentral-t search."""
    if mean_diff == 0:
        return None
    effect = abs(mean_diff) / sd_diff
    for n in range(4, 501):
        df = n - 1
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        ncp = effect * np.sqrt(n)
        power_n = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        if power_n >= power:
            return n
    return None


results = {}
for edge in ("whatsapp", "youtube"):
    full = np.array(R[edge]["full_system"]["adaptive"]["auc_per_fold"])
    base = np.array(R[edge]["snapshot_baseline"]["adaptive"]["auc_per_fold"])
    diffs = full - base

    lo, hi, mean_diff, sd_diff = cluster_bootstrap_ci(diffs)
    t_stat, p_value = paired_ttest(diffs)
    n_needed = n_folds_for_power(mean_diff, sd_diff)

    results[edge] = {
        "full_system_per_fold": full.tolist(),
        "baseline_per_fold": base.tolist(),
        "per_fold_diff": diffs.tolist(),
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "bootstrap_95ci_low": lo,
        "bootstrap_95ci_high": hi,
        "paired_t_stat": t_stat,
        "paired_t_pvalue": p_value,
        "n_folds_for_80pct_power": n_needed,
    }

print(json.dumps(results, indent=2))
