import copy
import json

from docx import Document
from docx.table import Table

PAPER = r"D:\DF paper\paper\Temporal-Drift-Custody-Integrity.docx"
doc = Document(PAPER)

anchor = None
for p in doc.paragraphs:
    if p.text.strip() == "8. Discussion":
        anchor = p
        break
assert anchor is not None

heading_ref = None
for p in doc.paragraphs:
    if p.text.strip().startswith("7.7 Cross-transform"):
        heading_ref = p
        break
assert heading_ref is not None

table_caption_ref = None
for p in doc.paragraphs:
    if p.text.strip().startswith("Table 13."):
        table_caption_ref = p
        break
assert table_caption_ref is not None

ref_table = doc.tables[12]  # Table 13, simple 2-col style


def insert_heading_before(anchor_para, text):
    p = anchor_para.insert_paragraph_before("")
    p.style = heading_ref.style
    p.add_run(text)
    return p


def insert_body_before(anchor_para, text):
    p = anchor_para.insert_paragraph_before("")
    p.add_run(text)
    return p


def insert_table_caption_before(anchor_para, text):
    p = anchor_para.insert_paragraph_before("")
    r = p.add_run(text)
    r.italic = True
    r.font.size = table_caption_ref.runs[0].font.size
    return p


def clone_table_before(anchor_para, header, rows):
    new_tbl = copy.deepcopy(ref_table._tbl)
    anchor_para._p.addprevious(new_tbl)
    t = Table(new_tbl, doc)
    n_cols = len(header)
    n_rows_needed = 1 + len(rows)

    # adjust column count by cloning/removing cells in each row (ref table has 2 cols)
    while len(t.columns) < n_cols:
        t.add_column(t.columns[0].width if t.columns else None)
    while len(t.columns) > n_cols:
        # remove last column's cells from each row via XML
        for row in t.rows:
            row._tr.remove(row._tr.tc_lst[-1])

    while len(t.rows) < n_rows_needed:
        new_row_el = copy.deepcopy(t.rows[-1]._tr)
        t.rows[-1]._tr.addnext(new_row_el)
    while len(t.rows) > n_rows_needed:
        last = t.rows[-1]._tr
        last.getparent().remove(last)

    for c_i, htext in enumerate(header):
        cell = t.rows[0].cells[c_i]
        for p in cell.paragraphs:
            for r in list(p.runs):
                r.text = ""
            if p.runs:
                p.runs[0].text = htext
            else:
                p.add_run(htext)

    for r_i, row_data in enumerate(rows, start=1):
        for c_i, val in enumerate(row_data):
            cell = t.rows[r_i].cells[c_i]
            for p in cell.paragraphs:
                for r in list(p.runs):
                    r.text = ""
                if p.runs:
                    p.runs[0].text = val
                else:
                    p.add_run(val)
    return t


# ================= 7.8 Post-hoc probability calibration =================
insert_heading_before(anchor, "7.8 Post-hoc probability calibration")

insert_body_before(
    anchor,
    "Section 7.4's Expected Calibration Error used a min-max-normalized raw score as a "
    "pseudo-probability, fit and evaluated on the same data, which the paper's own "
    "limitations list named as a real gap: a proper post-hoc calibration step, fit on a "
    "held-out split rather than the evaluation set itself, was named future work rather "
    "than something delivered. We close that gap here: for each edge type, the "
    "adaptive-tier full-system band-excess score pool (genuine plus synthetic attack "
    "instances) is split 60/40 into a train and a disjoint test set; Platt scaling "
    "(logistic regression on the raw score) and isotonic regression are each fit on the "
    "train split only, then applied to the test split, and ECE is computed on the "
    "resulting calibrated probabilities against the test labels the calibration step "
    "never saw."
)

cal = json.load(open(r"D:\DF paper\scripts\calibration_result.json"))
insert_table_caption_before(
    anchor,
    "Table 15. Post-hoc calibration, held-out train/test split: Expected Calibration Error "
    "before and after Platt scaling and isotonic regression, adaptive-tier full-system "
    "score, evaluated only on the test split.",
)
clone_table_before(
    anchor,
    header=["Edge", "Train N", "Test N", "Raw min-max ECE", "Platt-scaled ECE", "Isotonic ECE"],
    rows=[
        ["WhatsApp", str(cal["whatsapp"]["n_train"]), str(cal["whatsapp"]["n_test"]),
         f"{cal['whatsapp']['raw_minmax_ece']:.3f}", f"{cal['whatsapp']['platt_ece']:.3f}",
         f"{cal['whatsapp']['isotonic_ece']:.3f}"],
        ["YouTube", str(cal["youtube"]["n_train"]), str(cal["youtube"]["n_test"]),
         f"{cal['youtube']['raw_minmax_ece']:.3f}", f"{cal['youtube']['platt_ece']:.3f}",
         f"{cal['youtube']['isotonic_ece']:.3f}"],
    ],
)

insert_body_before(
    anchor,
    "On a genuinely held-out test split, raw min-max-normalized scores give an ECE of "
    "0.33 on both edges, close to the uncalibrated range Section 7.4 already reported. "
    "Platt scaling cuts this to 0.088 (WhatsApp) and 0.053 (YouTube); isotonic regression "
    "gives an almost identical result, 0.089 and 0.057. Both post-hoc methods reduce ECE "
    "by roughly a factor of four to six, a real, held-out-validated calibration "
    "improvement, not a same-data artifact. Platt scaling and isotonic land within a few "
    "thousandths of each other throughout, so there is no evidence either technique has a "
    "meaningful edge over the other here; either is a reasonable default. This closes the "
    "specific gap Section 7.4 and Section 8.4 named: a genuine post-hoc calibration step, "
    "evaluated honestly on data it never saw during fitting, is now reported rather than "
    "only proposed."
)

# ================= 7.9 False transform-type declaration robustness =================
insert_heading_before(anchor, "7.9 False transform-type declaration robustness")

insert_body_before(
    anchor,
    "Section 8.5 named an open question about the threat model: the per-edge drift model "
    "scores an edge's drift against the baseline for its declared transform type, so an "
    "attacker who declares a false transform type for a genuine edge, claiming a "
    "WhatsApp re-encode was a YouTube re-encode or vice versa, is scored against the "
    "wrong learned baseline. The paper's own hypothesis was that a false declaration "
    "might still surface as anomalous, since a genuine transform's drift often would not "
    "match a different type's baseline either, but this was explicitly flagged as "
    "untested. We test it directly here: every real genuine WhatsApp and YouTube delta "
    "from the 86-triplet set is scored twice, once against its true type's drift model "
    "and threshold (the honest case), and once against the other type's model and "
    "threshold (the falsely-declared case)."
)

fd = json.load(open(r"D:\DF paper\scripts\false_declaration_result.json"))
insert_table_caption_before(
    anchor,
    "Table 16. False transform-type declaration: flag rate when a genuine edge's true "
    "transform type is scored honestly versus scored against the other type's baseline "
    "(N = 86 genuine deltas per true type).",
)
clone_table_before(
    anchor,
    header=["True edge type", "Declared type", "Flag rate", "Mean score", "Threshold used"],
    rows=[
        ["WhatsApp (genuine)", "WhatsApp (honest)", f"{fd['whatsapp']['whatsapp']['flag_rate']:.1%}",
         f"{fd['whatsapp']['whatsapp']['mean_score']:.2f}", f"{fd['whatsapp']['whatsapp']['threshold']:.2f}"],
        ["WhatsApp (genuine)", "YouTube (false)", f"{fd['whatsapp']['youtube']['flag_rate']:.1%}",
         f"{fd['whatsapp']['youtube']['mean_score']:.2f}", f"{fd['whatsapp']['youtube']['threshold']:.2f}"],
        ["YouTube (genuine)", "YouTube (honest)", f"{fd['youtube']['youtube']['flag_rate']:.1%}",
         f"{fd['youtube']['youtube']['mean_score']:.2f}", f"{fd['youtube']['youtube']['threshold']:.2f}"],
        ["YouTube (genuine)", "WhatsApp (false)", f"{fd['youtube']['whatsapp']['flag_rate']:.1%}",
         f"{fd['youtube']['whatsapp']['mean_score']:.2f}", f"{fd['youtube']['whatsapp']['threshold']:.2f}"],
    ],
)

insert_body_before(
    anchor,
    "The result is asymmetric, and only partly reassuring. A genuine WhatsApp edge "
    "falsely declared as YouTube is flagged 34.9% of the time, well above the 10.5% "
    "baseline false-positive rate the honest declaration produces, so this direction of "
    "false declaration makes an edge more suspicious, not less, a genuine partial defense "
    "matching the paper's original hypothesis. The other direction does not: a genuine "
    "YouTube edge falsely declared as WhatsApp is flagged only 1.2% of the time, well "
    "below its own honest 10.5% baseline, meaning this specific false declaration is an "
    "effective evasion route, not a partial defense. The mechanism is calibration "
    "asymmetry rather than anything deeper: YouTube's threshold (-2.28) sits close to its "
    "own genuine mean (-3.56) relative to WhatsApp's threshold (3.03) sitting further from "
    "its own genuine mean (-0.56), so WhatsApp's baseline is calibrated more loosely "
    "relative to YouTube deltas than YouTube's baseline is relative to WhatsApp deltas. "
    "The practical conclusion is that false transform-type declaration is not a defense "
    "the system provides uniformly: it is a real, exploitable gap in one direction "
    "(YouTube edges relabeled as WhatsApp) and an accidental extra layer of protection in "
    "the other, and which direction an attacker would pick depends on which edge type "
    "they are trying to disguise."
)

# ================= 7.10 Multi-hop path-propagation robustness =================
insert_heading_before(anchor, "7.10 Multi-hop path-propagation robustness")

insert_body_before(
    anchor,
    "No evaluation reported so far in this paper exercises the multi-hop path-propagation "
    "component at all: every VISION triplet is a single hop (native to one re-encode), "
    "and Section 7.1's synthetic construction is also single-edge. The architecture's own "
    "motivation for path propagation is a trajectory-break attacker whose individual hops "
    "each look locally plausible but whose cumulative path is anomalous, a scenario a "
    "single-edge drift test cannot see by construction. We test this directly with a "
    "synthetic multi-hop construction built from real per-edge score statistics: for a "
    "chain of N hops of one transform type, genuine chains bootstrap-resample N real "
    "per-edge drift scores and sum them; a maximally cautious \"creeping\" attacker sets "
    "every hop exactly at, never over, its own edge-level 90th-percentile threshold, so a "
    "single-edge test never flags any individual hop, and asks whether the cumulative "
    "path-break z-test (Section 4's z = 3.0 threshold) catches the sum anyway."
)

pp = json.load(open(r"D:\DF paper\scripts\path_propagation_result.json"))
sweep_wa = pp["chain_length_sweep"]["whatsapp"]
sweep_yt = pp["chain_length_sweep"]["youtube"]
key_lengths = ["1", "3", "5", "7", "9", "15", "20"]

insert_table_caption_before(
    anchor,
    "Table 17. Maximal \"creeping\" attacker (every hop exactly at its own threshold) "
    "versus chain length N: z-score of the cumulative path sum and whether it clears the "
    "z = 3.0 path-break threshold, per transform type.",
)
clone_table_before(
    anchor,
    header=["N hops", "WhatsApp: attack z", "WhatsApp: flagged?", "YouTube: attack z", "YouTube: flagged?"],
    rows=[
        [n, f"{sweep_wa[n]['max_creeping_attack_z']:.2f}", "yes" if sweep_wa[n]["creeping_attack_flagged_at_z3"] else "no",
         f"{sweep_yt[n]['max_creeping_attack_z']:.2f}", "yes" if sweep_yt[n]["creeping_attack_flagged_at_z3"] else "no"]
        for n in key_lengths
    ],
)

wa_cross = pp["crossover_chain_length"]["whatsapp"]["analytic_min_hops"]
yt_cross = pp["crossover_chain_length"]["youtube"]["analytic_min_hops"]
insert_body_before(
    anchor,
    f"For N independent hops with per-hop mean mu, standard deviation sigma, and "
    f"threshold tau, the maximal creeping attacker's path sum is N*tau while the genuine "
    f"path sum has mean N*mu and standard deviation sigma*sqrt(N), so the attacker's "
    f"z-score grows as sqrt(N)*(tau-mu)/sigma: longer chains make the cumulative test "
    f"strictly more powerful against a per-hop-bounded attacker, confirmed empirically in "
    f"Table 17. For realistic short chains, 2 to 3 hops, matching what the real VISION "
    f"single-hop data and most of this paper's evaluation can test, the maximal creeping "
    f"attacker is never flagged on either edge type: path propagation provides no "
    f"protection at this length, and a trajectory-break attacker who stays within each "
    f"hop's own margin is indistinguishable from a single-edge drift-only system at N = 2 "
    f"or 3. The crossover point is transform-type-dependent and, for YouTube edges, "
    f"large: the analytic minimum chain length is {wa_cross:.1f} hops for WhatsApp-type "
    f"edges (matching the empirical result, flagged starting at N = 9) but "
    f"{yt_cross:.1f} hops for YouTube-type edges, since YouTube's threshold sits closer "
    f"to its own genuine mean relative to its variance than WhatsApp's does, the same "
    f"calibration asymmetry Section 7.9 found in the opposite context. Genuine chains are "
    f"flagged at a low, stable rate (roughly 1 to 3%) across every chain length tested, "
    f"so the z = 3.0 threshold is not simply too sensitive; it is specifically "
    f"under-powered against a maximally cautious multi-hop attacker at realistic chain "
    f"lengths. This is a genuine, previously untested limitation, not a reassurance: the "
    f"multi-hop path-propagation component, as calibrated, adds real protection only for "
    f"custody chains substantially longer than the two-to-three-hop chains this paper's "
    f"other evaluations use."
)

doc.save(PAPER)
print("Inserted sections 7.8, 7.9, 7.10.")
