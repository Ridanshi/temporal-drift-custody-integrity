from docx import Document

PAPER = r"D:\DF paper\paper\Temporal-Drift-Custody-Integrity.docx"
doc = Document(PAPER)
P = doc.paragraphs


def set_single_run(idx, text):
    p = P[idx]
    p.runs[0].text = text
    for r in p.runs[1:]:
        r.text = ""


def replace_sub(idx, old, new, run_idx=0):
    assert old in P[idx].runs[run_idx].text, f"substring not found in para {idx} run {run_idx}"
    P[idx].runs[run_idx].text = P[idx].runs[run_idx].text.replace(old, new)


# --- Para 78: headline result ---
set_single_run(78, (
    "On real data, the full system holds AUC \u2265 0.90 across every attack tier and every "
    "re-encode edge type, while the snapshot-fusion baseline degrades more gradually than the "
    "original 8-triplet pass suggested, from 0.91/0.91 on the naive tier to 0.86/0.85 on the "
    "adaptive tier, the same directional gap the paper's central claim predicts, though smaller "
    "in absolute terms at this larger, more representative N. Where 7.1's synthetic attacker "
    "produced a construction artifact, the real-data pass produces the effect the method was "
    "designed to demonstrate, evidence that the artifact in 7.1 lives in the synthetic attacker "
    "rather than in the underlying detector."
))

# --- Para 82: McNemar significance, full rewrite ---
set_single_run(82, (
    "At N = 86, four of the six comparisons remain significant well beyond p < 0.05: both "
    "edges' heuristic tier (p = 5.9e-4, 2.3e-4) and adaptive tier (p = 4.6e-3, 1.2e-3), where "
    "between 27 and 36 instances that the baseline misclassifies are correctly classified by "
    "the full system, against 9 to 10 instances in the opposite direction. The naive tier does "
    "not reach significance on either edge (p = 0.66 WhatsApp, p = 0.33 YouTube), even though "
    "the full system's naive-tier AUC (1.00) still exceeds the baseline's (0.91): at this "
    "larger N, the naive tier's discordant pairs are closer to balanced (12 vs. 9 on WhatsApp, "
    "16 vs. 10 on YouTube) than the original 8-triplet pass suggested, where every discordant "
    "pair broke the same direction. This is the same pattern already visible in Section 7.2.4's "
    "held-out validation: small-N significance results can be more extreme than the underlying "
    "effect, and a larger sample gives a more honest, more modest picture on a tier where both "
    "methods are already near-ceiling. The heuristic- and adaptive-tier results, where the AUC "
    "gap is real and substantial, remain strong evidence that the improvement over snapshot "
    "fusion on the tiers that matter most is not an artifact of a favorable evaluation split."
))

# --- Para 84: ablation numbers ---
set_single_run(84, (
    "The co-evasion-only ablation's collapse on the adaptive tier, 0.45/0.45, below chance but "
    "less severely than the original 8-triplet pass (0.42/0.42), combined with the full system "
    "staying strong at 0.90/0.92, shows the drift component and the co-evasion component are "
    "catching different attacks rather than redundant ones. The adaptive attacker in this "
    "evaluation is specifically constructed to target the co-evasion test's normal band "
    "(Section 3, Tier 3), so a below-chance co-evasion-only score on that tier is the expected "
    "signature of a successful targeted evasion of that one component; what matters is that the "
    "drift-only ablation remains strong (0.95/0.94) on the same tier, and the combination "
    "recovers to 0.90/0.92 rather than inheriting the co-evasion component's failure. This is "
    "the ablation behaving exactly as the two-component design intends: an attacker who defeats "
    "one channel is still caught by the other."
))

# --- Para 112: learned-baseline results, full rewrite ---
set_single_run(112, (
    "Isolation Forest and One-Class SVM both range across a believable middle band at this "
    "larger N, no longer the suspicious perfect-or-unstable pattern the original 8-point pass "
    "produced: Isolation Forest from 0.625 (YouTube adaptive) to 0.994 (YouTube naive), "
    "One-Class SVM from 0.848 (YouTube heuristic) to 1.00 (YouTube naive). Section 8.3 "
    "discusses this shift directly: the original pass's perfect 1.00 One-Class SVM score and "
    "Isolation Forest's collapse to 0.13, below chance, were flagged there as likely "
    "small-sample artifacts of fitting flexible models to eight points and testing on those "
    "same eight; this larger-N rerun is the direct empirical confirmation of that prediction, "
    "not a new finding requiring separate interpretation."
))

# --- Para 118: calibration, full rewrite ---
set_single_run(118, (
    "The full system's calibration is modestly better than the baseline's on both edge types, "
    "and both improve substantially at this larger N relative to the original 8-point pass: "
    "ECE around 0.20 to 0.28 here versus 0.6 to 0.7 there, consistent with ECE being a "
    "notoriously noisy, sample-size-sensitive estimator at small N rather than evidence the "
    "underlying scores themselves became better calibrated. Neither method is well calibrated "
    "in an absolute sense even at this improved level; an ECE around 0.2 still indicates the "
    "raw anomaly scores, min-max normalized, are an imperfect stand-in for a true probability. "
    "We report this honestly rather than omitting it: our band-excess and drift-excess scores "
    "were designed for ranking (AUC), not for probability calibration, and Section 8.4 names "
    "proper probabilistic calibration, for instance via Platt scaling or isotonic regression "
    "fit on a held-out genuine set, as a concrete direction the modest improvement here does "
    "not yet deliver on its own."
))

# --- Para 124: threshold sensitivity ---
set_single_run(124, (
    "AUC varies by at most 0.044 across a nearly 20-percentile-point sweep, still a narrow "
    "band though wider than the 0.013 spread the original 8-triplet pass showed, consistent "
    "with a larger, more representative sample revealing slightly more real threshold-"
    "sensitivity than a small sample could show. This remains a reassuring result: it "
    "indicates the headline numbers in Table 3 are not an artifact of a fortuitously chosen "
    "threshold, which is a real risk with any percentile-calibrated method and a common "
    "failure mode this sweep is specifically designed to catch."
))

# --- Para 134: cross-transform, full rewrite (honest, no longer "inconclusive") ---
set_single_run(134, (
    "Contrary to our design hypothesis, the cross-fit model again scores higher, not lower, "
    "than the matched model, and now on a substantially larger sample: 86 genuine deltas per "
    "transform type rather than 8. At N = 8, we read this as two noisy small-sample fits "
    "differing by less than their own sampling variance, and declined to draw a conclusion "
    "either way. At N = 86, the same direction persisting is harder to dismiss as noise, and "
    "we no longer think \u201cinconclusive\u201d is the honest label. What this result does and does "
    "not show: it is one specific comparison, a WhatsApp-fit model scoring YouTube data on the "
    "adaptive tier, not a full bidirectional test of whether per-transform-type baselines "
    "matter across the whole system; the drift-only and full-system results elsewhere in this "
    "paper still use the transform-matched models specified in Section 4.2, and this ablation "
    "does not by itself show those numbers would hold, improve, or degrade under a pooled "
    "single-model design. What it does show is that the specific design assumption in Section "
    "4.2, that a transform-matched model necessarily outperforms a mismatched one, is not "
    "supported by this test at either sample size available, and a pooled or otherwise unified "
    "drift model is a legitimate alternative worth testing directly in future work rather than "
    "a hypothesis we can continue to treat as settled by construction."
))

# --- Table 13 caption (para 131) ---
set_single_run(131, (
    "Table 13. Cross-transform generalization test on the adaptive tier: at N = 86 genuine "
    "deltas per transform type, a model fit on the wrong transform type still outperforms the "
    "matched model."
))

# --- Table 11 caption (para 121) ---
set_single_run(121, (
    "Table 11. Full-system adaptive-tier AUC across calibration percentiles 80 to 99. AUC "
    "varies by at most 0.044 across the entire sweep."
))

# --- Para 141: learned-baseline discussion in 8.3, full rewrite ---
set_single_run(141, (
    "Table 9's original 8-point pass produced a perfect 1.00 One-Class SVM AUC on every "
    "attack tier and edge type, and we argued at the time that this deserved scrutiny rather "
    "than celebration: a flexible RBF-kernel boundary fit to eight points and tested on those "
    "same eight points has more than enough capacity to wrap tightly around them, and "
    "\"perfect\" separation on data the model was fit to says nothing about a held-out case. "
    "Isolation Forest's instability in the same original pass, from 0.92 down to 0.13, below "
    "chance, was the same warning sign in its other common form, a fitted model whose behavior "
    "swings wildly with the specific random splits it happens to draw at small N. Rerunning "
    "both on the full 86-triplet subset confirms this reading directly: One-Class SVM now "
    "ranges from 0.848 to 1.00 and Isolation Forest from 0.625 to 0.994, a believable middle "
    "band in place of the earlier perfect-or-unstable pattern. Neither result should be read "
    "as \"learned detectors beat our interpretable one\" or the reverse; both the original small-"
    "N pattern and its resolution at larger N are evidence that N = 8 was too small to compare "
    "any two fitted models meaningfully, learned or statistical, exactly the concrete "
    "prediction Section 7.2's held-out-split limitation made in the abstract. The XAI "
    "literature in Section 2.6 [34-38] frames black-box detectors as needing post-hoc "
    "explanation to be trusted; here we would add a second reason for caution specific to "
    "small forensic samples, a black-box detector's confident-looking score at small N can be "
    "confidently wrong in a way that is not visible from the score alone, while our drift and "
    "co-evasion scores at least expose the quantity, a log-likelihood or a distance to a "
    "calibrated band, that produced them, and the worked case study in Section 7.6 shows that "
    "quantity is inspectable by a human reviewer rather than only by an auditor of the fitting "
    "procedure."
))

doc.save(PAPER)
print("Saved core N=86 prose/table-caption updates.")
