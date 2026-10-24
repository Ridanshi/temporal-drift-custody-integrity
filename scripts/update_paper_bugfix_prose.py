from docx import Document

PAPER = r"D:\DF paper\paper\Temporal-Drift-Custody-Integrity.docx"
doc = Document(PAPER)
P = doc.paragraphs


def set_single_run(idx, text):
    p = P[idx]
    p.runs[0].text = text
    for r in p.runs[1:]:
        r.text = ""


# --- Para 82: McNemar prose, corrected numbers ---
set_single_run(82, (
    "At N = 86, four of the six comparisons remain significant well beyond p < 0.05: both "
    "edges' heuristic tier (p = 4.3e-4, 2.5e-4) and adaptive tier (p = 3.6e-3, 1.4e-3), where "
    "between 26 and 34 instances that the baseline misclassifies are correctly classified by "
    "the full system, against 8 to 9 instances in the opposite direction. The naive tier does "
    "not reach significance on either edge (p = 0.65 WhatsApp, p = 0.40 YouTube), even though "
    "the full system's naive-tier AUC (1.00) still exceeds the baseline's (0.91): at this "
    "larger N, the naive tier's discordant pairs are closer to balanced (11 vs. 8 on WhatsApp, "
    "14 vs. 9 on YouTube) than the original 8-triplet pass suggested, where every discordant "
    "pair broke the same direction. This is the same pattern already visible in Section 7.2.4's "
    "held-out validation: small-N significance results can be more extreme than the underlying "
    "effect, and a larger sample gives a more honest, more modest picture on a tier where both "
    "methods are already near-ceiling. The heuristic- and adaptive-tier results, where the AUC "
    "gap is real and substantial, remain strong evidence that the improvement over snapshot "
    "fusion on the tiers that matter most is not an artifact of a favorable evaluation split."
))

# --- Para 100: DFD comparison, corrected ---
set_single_run(100, (
    "The real and synthetic naive-tier numbers mostly agree: the baseline and drift-only "
    "components differ by 0.06 or less between the two columns, and the ordering is nearly "
    "identical, drift-only and full system lead, baseline trails behind. The co-evasion-only "
    "ablation is close between the two columns too, 0.79/0.79 synthetic against 0.75/0.75 "
    "real, a small 0.04 gap consistent with the agreement seen elsewhere in the table. The "
    "full system itself shows a modest real-data dip on the WhatsApp edge specifically, 1.00 "
    "synthetic against 0.94 real, while the YouTube edge stays essentially perfect on both "
    "(1.00 synthetic, 1.00 real); a real face-swap is slightly harder for the full system to "
    "catch on one edge than the synthetic construction predicted, though 0.94 remains a strong "
    "result. The baseline itself scores higher on real deepfake tamper (0.94/0.97) than on the "
    "synthetic naive attacker (0.91/0.91); a real face-swap apparently produces a more "
    "detectable manifest-intrinsic conflict than a magnitude-5 Gaussian perturbation does for "
    "the baseline specifically, which if anything makes the synthetic naive tier the more "
    "conservative, harder construction for that one method, not an inflated one. This is a "
    "direct answer to the concern raised in Section 8.2 about whether the synthetic attacker's "
    "numbers generalize: for the one tier a real dataset can actually test, the full system's "
    "own result mostly does, with the one WhatsApp-edge exception noted above. It does not "
    "touch Section 7.2.4's open question, whether the co-evasion test's adaptive-tier "
    "advantage holds under held-out scoring, since DFD cannot supply a real adaptive attacker "
    "against this detector; that gap is unaffected by this result and remains exactly as open "
    "as Section 7.2.4 leaves it."
))

# --- Para 106: Celeb-DF comparison, corrected (major rewrite) ---
set_single_run(106, (
    "The full system and drift-only ablation both hold perfect AUC on real data, matching the "
    "synthetic construction's direction and confirming the headline result is not sensitive to "
    "whether the heuristic-tier attack is synthetic or real. The baseline does about as well on "
    "real heuristic-tier data (0.95/0.97) as the synthetic construction predicted (0.86/0.86), a "
    "modestly stronger manifest-intrinsic conflict signature from a real deepfake plus standard "
    "re-encode than the synthetic perturbation modeled. The one large divergence is the "
    "co-evasion-only ablation, which scores 0.94/0.94 on the synthetic heuristic attacker but "
    "collapses to 0.50/0.50, chance level, on real Celeb-DF content, a 0.44 gap far outside the "
    "agreement seen everywhere else in the table. Read alone this looks like a serious weakness "
    "in the co-evasion component specifically for real heuristic-tier tampering; read alongside "
    "drift-only holding perfect AUC on the same real data, the finding is narrower and more "
    "precise than that: on real Celeb-DF heuristic-tier content, the full system's near-perfect "
    "detection is being carried entirely by the drift component, and the co-evasion component "
    "is contributing essentially nothing, not a small amount, to that specific real-data "
    "result. This is a materially different picture from the synthetic construction, where "
    "co-evasion-only still tracks well above chance (0.94) on the same tier, and it means the "
    "synthetic heuristic-tier attacker is not a good stand-in for how the co-evasion test "
    "specifically behaves against a real off-the-shelf re-encode of real deepfake content, even "
    "though the full system's aggregate number is unaffected because the drift component alone "
    "already solves this tier. Which component carries a given tier is not fixed across the "
    "paper: Section 7.2.3's real-data result found co-evasion carrying real weight on the "
    "adaptive tier, while this result finds it carrying essentially none on the real heuristic "
    "tier; both are evidence that the two components' relative contributions are tier- and "
    "data-source-dependent, which is itself consistent with the paper's central ablation "
    "argument that no single channel needs to be reliable on every tier, only the combination "
    "does."
))

# --- Para 128: worked case study, corrected numbers ---
set_single_run(128, (
    "The codec-fingerprint and frame-count components of the feature vector are identical "
    "before and after the WhatsApp re-encode in this instance, only the compression-artifact "
    "and noise-residual components shift, and both shifts are small enough that the drift "
    "score sits well below threshold. The co-evasion distance, 1.942, sits comfortably inside "
    "the normal band, 0.591 to 3.296, correctly reflecting that a genuine platform re-encode "
    "does perturb the manifest-intrinsic relationship somewhat, without crossing into either "
    "tail. This triplet was part of the same 86 genuine triplets used to fit the models "
    "(Section 7.2's caveat), so an in-band, not-flagged verdict here is expected rather than a "
    "held-out validation; we include it to make the pipeline's actual arithmetic legible, not "
    "as independent evidence."
))

# --- Para 153: DFD/Celeb-DF limitations bullet, corrected ---
set_single_run(153, (
    "Attack instances in both Section 7.1 and 7.2 are statistically simulated rather than "
    "drawn from a real tampered-evidence corpus. Section 7.2.5 closes part of this gap with a "
    "real DFD deepfake-tamper naive-tier validation: the baseline and drift-based components "
    "closely match Table 3's synthetic naive-tier numbers, within 0.06 AUC, and the "
    "co-evasion-only ablation now also agrees closely (0.75 real vs. 0.79 synthetic); the full "
    "system shows a modest real-data dip on the WhatsApp edge specifically (0.94 vs. 1.00 "
    "synthetic) while staying essentially perfect on YouTube. Section 7.2.6 closes another part "
    "with a real Celeb-DF deepfake-plus-standard-re-encode heuristic-tier validation: "
    "drift-only and the full system both hold perfect AUC on real data, matching the synthetic "
    "direction, but the co-evasion-only ablation collapses to chance (0.50) on real content "
    "against 0.94 synthetic, a genuine, large divergence unlike anything seen in the DFD "
    "naive-tier pass. This is a real finding: it shows the co-evasion component contributes "
    "essentially nothing to the full system's real heuristic-tier result, which is instead "
    "carried entirely by the drift component, a materially different picture from the "
    "synthetic construction where co-evasion-only still tracks well above chance on the same "
    "tier. Tier 3 alone remains synthetic: it requires an attacker who deliberately forges "
    "metadata against this specific detector, which by definition no pre-existing real dataset "
    "can provide, real or simulated adaptive-attacker data against this exact system is the "
    "only way to close what remains of this gap, and that is a live red-team exercise, not a "
    "download."
))

doc.save(PAPER)
print("Saved all prose corrections.")
