"""Two robustness checks on the Section 7.2.4 leave-one-device-out adaptive-tier
mean-difference test (Table 6), using the same 15 held-out folds, no new data.

Table 6's paired t-test assumes normally distributed paired differences. The paper's
own text (Section 7.2.4) already names why that assumption is shaky here: the
snapshot-fusion baseline's per-device behavior is bimodal, a few large collapses to
near-chance mixed with many near-ceiling folds, not a smooth normal spread. Two
distribution-free alternatives on the exact same 15 paired differences:

1. Wilcoxon signed-rank test: rank-based, robust to the outlier-heavy shape, standard
   alternative to the paired t-test for small N with skew/heavy tails.
2. Exact sign-flip permutation test: under the null the sign of each device's paired
   difference is an independent coin flip; with N=15 all 2**15 = 32768 sign patterns
   are enumerable exactly, no asymptotic approximation needed at all.

This does not manufacture significance, it tests whether the t-test's p-value is a
test-choice artifact of the normality assumption or the real answer; report whichever
way each one lands.
"""
import json
import itertools

import numpy as np
from scipy import stats

with open(r"D:\DF paper\scripts\held_out_eval_n15_result.json") as f:
    R = json.load(f)


def exact_sign_permutation_pvalue(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    observed = abs(diffs.mean())
    count_as_extreme = 0
    total = 2 ** n
    for signs in itertools.product([1, -1], repeat=n):
        signed_mean = abs((np.array(signs) * diffs).mean())
        if signed_mean >= observed - 1e-12:
            count_as_extreme += 1
    return count_as_extreme / total


results = {}
for edge in ("whatsapp", "youtube"):
    full = np.array(R[edge]["full_system"]["adaptive"]["auc_per_fold"])
    base = np.array(R[edge]["snapshot_baseline"]["adaptive"]["auc_per_fold"])
    diffs = full - base

    w_stat, w_pvalue = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
    perm_pvalue = exact_sign_permutation_pvalue(diffs)

    results[edge] = {
        "n": len(diffs),
        "mean_diff": round(float(diffs.mean()), 4),
        "wilcoxon_stat": round(float(w_stat), 3),
        "wilcoxon_pvalue": round(float(w_pvalue), 4),
        "exact_permutation_pvalue": round(float(perm_pvalue), 4),
    }

print(json.dumps(results, indent=2))
with open(r"D:\DF paper\scripts\robustness_tests_n15_result.json", "w") as f:
    json.dump(results, f, indent=2)
