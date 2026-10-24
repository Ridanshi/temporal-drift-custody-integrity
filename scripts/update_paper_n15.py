from docx import Document

PAPER = r"D:\DF paper\paper\Temporal-Drift-Custody-Integrity.docx"
doc = Document(PAPER)
P = doc.paragraphs

# --- Para 1: working paper note ---
P[1].runs[0].text = (
    "Working paper draft: reference implementation validated on synthetic data and a "
    "15-device real-data subset (VISION dataset) with leave-one-device-out held-out "
    "validation; scaling to the full VISION corpus remains future work."
)

# --- Para 3: abstract, replace held-out sentence only ---
old_sentence = (
    "A subsequent leave-one-device-out held-out validation confirms this single-split "
    "result on the naive and heuristic tiers but not on the adaptive tier, where the "
    "baseline's held-out AUC (0.75) exceeds the full system's (0.57 to 0.62), though both "
    "estimates carry wide fold-to-fold variance at this sample size (Section 7.2.4)."
)
new_sentence = (
    "A subsequent leave-one-device-out held-out validation across 15 devices confirms "
    "this single-split result on every attack tier: on the adaptive tier specifically, the "
    "tier the co-evasion test exists to catch, the full system's mean held-out AUC (0.90 "
    "WhatsApp, 0.93 YouTube) exceeds the baseline's (0.85, 0.76) and never drops below 0.73 "
    "on any held-out device, while the baseline collapses to near-chance AUC on 3 to 4 of "
    "the 15 devices; the paired mean-difference test does not yet reach significance "
    "(p = 0.10 to 0.37), and a power analysis puts the fold count needed at 41 to 141 "
    "(Section 7.2.4)."
)
assert old_sentence in P[3].runs[0].text, "abstract sentence not found verbatim"
P[3].runs[0].text = P[3].runs[0].text.replace(old_sentence, new_sentence)

# --- Para 87: methodology ---
P[87].runs[0].text = (
    "Section 7.2.3's caveat is unambiguous: the eight genuine triplets used to fit the "
    "drift model and the co-evasion joint distribution are the same eight scored as the "
    "genuine class in Table 3, so the reported AUCs are optimistically biased by an unknown "
    "amount. We address this directly with leave-one-device-out cross-validation, scaled "
    "from an initial 4-device pass to the fifteen devices with usable triplets now "
    "available: for each device, every model is fit only on the other fourteen devices' "
    "genuine triplets, then scored against that device's own held-out genuine triplets plus "
    "synthetic attack instances anchored at the training-fold genuine mean. No genuine "
    "sample is ever scored by a model that was fit on it. AUC is reported as the mean and "
    "standard deviation across the fifteen folds, since raw scores from fifteen "
    "separately-fit GMMs are not guaranteed to sit on a comparable scale; F1 pools binary "
    "predictions across folds directly, which carries no such assumption."
)

# --- Para 88: Table 5 caption ---
P[88].runs[0].text = (
    "Table 5. Leave-one-device-out held-out validation: AUC (mean \u00b1 std across 15 "
    "folds) by method and attack tier. Each cell reports WhatsApp-edge / YouTube-edge."
)

# --- Para 90: Figure 6 caption ---
P[90].runs[0].text = (
    "Figure 6. Leave-one-device-out held-out validation, AUC by method and attack tier, "
    "error bars show \u00b1 1 std across the 15 held-out folds (Table 5)."
)

# --- Para 91: results prose, full rewrite ---
P[91].runs[0].text = (
    "On the naive and heuristic tiers, the held-out result confirms Table 3's direction: "
    "drift-only and full-system mean AUC (0.90 to 1.00) stay well clear of the baseline's "
    "(0.76 to 0.90), with error bars that mostly do not overlap. On the adaptive tier, the "
    "tier the co-evasion test exists specifically to catch, the held-out result now also "
    "confirms Table 3's direction on average: the full system's mean adaptive AUC, 0.90 on "
    "WhatsApp and 0.92 on YouTube, exceeds the baseline's, 0.85 and 0.76, and the full "
    "system never drops below 0.73 on any of the fifteen held-out devices. The baseline's "
    "own behavior is the more striking finding: it is bimodal rather than uniformly weaker, "
    "scoring a perfect 1.00 on most devices but collapsing to near-chance, as low as 0.15 on "
    "WhatsApp and 0.00 on YouTube, on three to four specific devices. The full system shows "
    "no such collapse on any device. This is a robustness claim, not yet a formally "
    "significant mean-difference claim; Table 6's paired test on the same fifteen folds "
    "does not reach p < 0.05 for either edge, because the baseline's bimodal variance "
    "dominates the paired difference's own variance. We report both readings directly: the "
    "full system is, on this evidence, both higher on average and categorically more "
    "reliable across device types than the baseline on the tier that matters most, but the "
    "specific claim of a statistically significant mean-AUC improvement is not yet "
    "established at this sample size, and the power analysis in Table 6 gives the honest "
    "target for when it would be."
)

# --- Para 92: minor fix ---
old92 = "is the simplest honest description of four numbers, but it is not a hypothesis test."
new92 = "is the simplest honest description of these per-fold numbers, but it is not a hypothesis test."
assert old92 in P[92].runs[0].text
P[92].runs[0].text = P[92].runs[0].text.replace(old92, new92)

# --- Para 94: bootstrap/power results, full rewrite ---
P[94].runs[0].text = (
    "The picture at fifteen folds is different from the four-fold pass in an instructive "
    "way. The mean per-fold difference (full system minus baseline) is now positive on both "
    "edges, +0.052 on WhatsApp and +0.169 on YouTube, the correct sign for the paper's "
    "central claim. The bootstrap interval excludes zero on YouTube ([0.004, 0.364]) but "
    "includes it on WhatsApp ([\u22120.044, 0.165]); the paired t-test does not reach "
    "significance on either edge (p = 0.372 WhatsApp, p = 0.101 YouTube). This is not a "
    "contradiction of the positive mean: it is a direct consequence of the baseline's "
    "bimodal per-device behavior described above, a distribution with a few large negative "
    "outliers and many near-zero or positive folds has high variance relative to its mean, "
    "which is exactly what a paired test penalizes. The power analysis reflects this "
    "honestly: reaching 80% power for the mean-difference test now requires 141 folds on "
    "WhatsApp and 41 on YouTube, larger than the 8 and 29 estimated from the original "
    "four-fold pass, because that small-sample estimate itself understated the true "
    "fold-to-fold variance now visible at N = 15. We do not read this as a weaker result "
    "than the four-fold pass, the sign flip from a baseline-favoring to a "
    "full-system-favoring mean difference, and the elimination of any full-system collapse "
    "on any held-out device, are real improvements, but we report the significance target "
    "exactly as the data gives it rather than rounding a favorable mean up to a proven claim."
)

# --- Para 152: limitations bullet ---
P[152].runs[0].text = (
    "Section 7.2's real-data results (Table 3) use only 8 genuine triplets (4 devices, 2 "
    "scenes) and originally reused the fitting set as the genuine test class rather than a "
    "held-out split; Section 7.2.4's leave-one-device-out validation, now run across "
    "fifteen devices, removes that specific reuse bias and confirms Table 3's direction on "
    "every tier including the adaptive tier, where it previously did not: the full system's "
    "mean held-out adaptive AUC now exceeds the baseline's on both edge types and never "
    "collapses on any single device, while the baseline itself collapses to near-chance on "
    "three to four of the fifteen. The specific claim that does not yet clear is "
    "statistical significance of the mean-AUC difference, which the paired test and power "
    "analysis in Table 6 report honestly (41 to 141 folds needed for 80% power, more than "
    "the fifteen available). This remains a meaningful limitation, but a narrower one than "
    "before: the direction and robustness of the effect are now supported by held-out data, "
    "only the formal significance test on the mean is not. Section 7.3's learned-baseline "
    "comparison (discussed in Section 8.3) and Section 7.7's cross-transform test are both "
    "still built on the original 8-triplet single-split subset; the expanded held-out split "
    "does not yet extend to them."
)

# --- Para 160: conclusion, replace the held-out clause ---
old160 = (
    "a leave-one-device-out held-out validation confirms this on the naive and heuristic "
    "tiers but not on the adaptive tier, where the effect reverses in point estimate within "
    "noise too large to call it conclusive, the single most consequential correction this "
    "paper makes to its own headline claim."
)
new160 = (
    "a leave-one-device-out held-out validation, now run across fifteen devices, confirms "
    "this on every tier including the adaptive tier: the full system's mean adaptive AUC "
    "exceeds the baseline's and never collapses on any held-out device, while the baseline "
    "itself collapses to near-chance on three to four of the fifteen, though the paired "
    "significance test on the mean difference does not yet clear at this sample size."
)
assert old160 in P[160].runs[0].text
P[160].runs[0].text = P[160].runs[0].text.replace(old160, new160)

old160b = (
    "The remaining work is scaling this real-data pass, more devices, a proper held-out "
    "genuine/test split, and a real rather than synthetic tampered-evidence corpus, to move "
    "from the proof of concept reported here to a publication-grade claim."
)
new160b = (
    "The remaining work is scaling further, beyond fifteen devices to the fold counts the "
    "power analysis specifies for formal significance on the adaptive-tier mean difference, "
    "and a real rather than synthetic tampered-evidence corpus for the adaptive tier "
    "specifically, to move from the proof of concept reported here to a publication-grade "
    "claim."
)
assert old160b in P[160].runs[0].text
P[160].runs[0].text = P[160].runs[0].text.replace(old160b, new160b)

# --- Table 5 (doc.tables[4]): update AUC cells ---
t5 = doc.tables[4]
table5_data = {
    "Snapshot-fusion baseline": ["0.90 \u00b1 0.18 / 0.88 \u00b1 0.28", "0.85 \u00b1 0.25 / 0.76 \u00b1 0.39", "0.85 \u00b1 0.26 / 0.76 \u00b1 0.40"],
    "Drift-only (ablation)": ["1.00 \u00b1 0.00 / 1.00 \u00b1 0.00", "1.00 \u00b1 0.00 / 1.00 \u00b1 0.01", "0.95 \u00b1 0.07 / 0.94 \u00b1 0.05"],
    "Co-evasion-only (ablation)": ["0.72 \u00b1 0.07 / 0.72 \u00b1 0.07", "0.90 \u00b1 0.13 / 0.90 \u00b1 0.13", "0.39 \u00b1 0.12 / 0.39 \u00b1 0.12"],
    "Full system": ["1.00 \u00b1 0.00 / 1.00 \u00b1 0.00", "1.00 \u00b1 0.00 / 1.00 \u00b1 0.00", "0.90 \u00b1 0.08 / 0.92 \u00b1 0.06"],
}
for row in t5.rows[1:]:
    method = row.cells[0].text.strip()
    if method in table5_data:
        vals = table5_data[method]
        for ci, val in enumerate(vals, start=1):
            cell = row.cells[ci]
            for p in cell.paragraphs:
                for r in list(p.runs):
                    r.text = ""
                if p.runs:
                    p.runs[0].text = val
                else:
                    p.add_run(val)
    else:
        print("WARNING: unmatched method row in Table 5:", method)

# --- Table 6 (doc.tables[5]): update comparison stats ---
t6 = doc.tables[5]
table6_data = {
    "WhatsApp": ["+0.052", "[-0.044, 0.165]", "0.372", "141"],
    "YouTube": ["+0.169", "[0.004, 0.364]", "0.101", "41"],
}
for row in t6.rows[1:]:
    edge = row.cells[0].text.strip()
    if edge in table6_data:
        vals = table6_data[edge]
        for ci, val in enumerate(vals, start=1):
            cell = row.cells[ci]
            for p in cell.paragraphs:
                for r in list(p.runs):
                    r.text = ""
                if p.runs:
                    p.runs[0].text = val
                else:
                    p.add_run(val)
    else:
        print("WARNING: unmatched edge row in Table 6:", edge)

doc.save(PAPER)
print("Saved. All assertions passed.")
