from docx import Document

PAPER = r"D:\DF paper\paper\Temporal-Drift-Custody-Integrity.docx"
doc = Document(PAPER)
P = doc.paragraphs


def set_single_run(idx, text):
    p = P[idx]
    p.runs[0].text = text
    for r in p.runs[1:]:
        r.text = ""


set_single_run(167, (
    "Section 7.4's calibration improvement over the baseline is real (ECE 0.203 vs. 0.269 "
    "on WhatsApp, 0.202 vs. 0.276 on YouTube at N = 86, both substantially better than the "
    "0.6 to 0.7 the original 8-point pass showed, consistent with ECE being a noisy, "
    "sample-size-sensitive estimator rather than evidence the scores themselves changed), "
    "and neither method is well calibrated in an absolute sense on that raw min-max score; "
    "the band-excess and drift-excess scores were designed for ranking, not probability "
    "calibration. A proper post-hoc calibration step is now delivered rather than only "
    "proposed: Section 7.8 fits Platt scaling and isotonic regression on a held-out train "
    "split and evaluates ECE only on a disjoint test split it never saw during fitting, "
    "cutting ECE to 0.05 to 0.09, roughly a four-to-six-fold improvement over the raw "
    "min-max score, with no evidence either calibration method has a meaningful edge over "
    "the other."
))

set_single_run(170, (
    "The per-edge drift model scores an edge's drift against the baseline for its declared "
    "transform type; Section 7.9 tests directly whether this defends against an attacker "
    "who declares a false transform type for a genuine edge, and finds an asymmetric "
    "answer rather than a clean one: a WhatsApp edge falsely declared as YouTube is "
    "flagged more often than if declared honestly, a real partial defense matching the "
    "paper's original hypothesis, but a YouTube edge falsely declared as WhatsApp is "
    "flagged far less often than if declared honestly, a real, exploitable evasion route "
    "in that direction specifically, traced to the two transform types' different "
    "threshold-to-mean calibration margins rather than anything deeper. Multi-hop path "
    "propagation, the paper's other defense against graph-structure attacks, is "
    "characterized directly in Section 7.10 rather than left untested: it provides no "
    "protection against a maximally cautious trajectory-break attacker at the two-to-three-"
    "hop chain lengths this paper's other evaluations use, and only becomes effective at "
    "chain lengths of 9 or more hops for WhatsApp-type edges and 60 or more for "
    "YouTube-type edges. Robustness to graph-structure attacks is therefore no longer an "
    "untested open question; it is a characterized, real, and partly negative result, both "
    "false declaration and short-chain trajectory breaks are exploitable given what this "
    "paper's own experiments now show, and hidden-hop omission specifically remains "
    "untested."
))

doc.save(PAPER)
print("Updated limitations bullets 167 and 170.")
