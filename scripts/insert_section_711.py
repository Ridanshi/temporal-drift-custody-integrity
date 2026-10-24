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
    if p.text.strip().startswith("7.10 Multi-hop"):
        heading_ref = p
        break
assert heading_ref is not None

table_caption_ref = None
for p in doc.paragraphs:
    if p.text.strip().startswith("Table 17."):
        table_caption_ref = p
        break
assert table_caption_ref is not None

ref_table = doc.tables[15]  # Table 17, 5 columns, exact match for the new table's shape


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

    while len(t.columns) < n_cols:
        t.add_column(t.columns[0].width if t.columns else None)
    while len(t.columns) > n_cols:
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


pe = json.load(open(r"D:\DF paper\scripts\path_elevation_result.json"))

# ================= 7.11 A rank-based path-anomaly signal =================
insert_heading_before(anchor, "7.11 A rank-based path-anomaly signal")

insert_body_before(
    anchor,
    "Section 7.10's sum-based z-test has a specific, structural weakness against the "
    "maximal creeping attacker: for N hops each held exactly at its own edge threshold, "
    "the attacker's best-case path sum grows linearly with N, but genuine path-sum "
    "variance grows only as the square root of N, so at short chain lengths the two "
    "distributions genuinely overlap, no choice of z-threshold separates them, because the "
    "attacker's maximum possible signal is itself close to ordinary genuine noise at "
    "small N. Retuning the threshold cannot fix a problem in the test statistic itself. "
    "We propose a different statistic, targeted specifically at the attacker's actual "
    "behavior, staying consistently near the edge of normal on every hop, rather than at "
    "the size of the accumulated total: count how many hops in a chain exceed an "
    "\"elevated\" percentile of their own transform type's genuine score distribution "
    "(a bar set below the edge-level flagging threshold itself, so this is a separate "
    "signal from single-edge drift detection), and test whether that count is improbable "
    "under a genuine chain, where each hop independently has a fixed, known probability of "
    "landing above that bar by construction. This is a one-sided binomial test, not a "
    "sum, so its power against a same-direction, every-hop attacker does not depend on the "
    "attacker's total accumulated deviation staying small relative to chain-level noise, "
    "only on how many hops are simultaneously elevated."
)

insert_table_caption_before(
    anchor,
    "Table 18. The elevated-hop-count test versus the sum-based z-test, same maximal "
    "creeping attacker as Table 17, elevated bar set at the 70th percentile of each "
    "transform type's genuine score distribution; genuine false-positive rate in "
    "parentheses.",
)
rows = []
for n in ["1", "2", "3", "4", "5", "7", "9", "11", "15", "20"]:
    wa_z = pe["whatsapp"]["z_test"][n]
    wa_e = pe["whatsapp"]["elevated_pctile_0.7"][n]
    yt_z = pe["youtube"]["z_test"][n]
    yt_e = pe["youtube"]["elevated_pctile_0.7"][n]
    rows.append([
        n,
        f"{'yes' if wa_z['attack_flagged'] else 'no'} ({wa_z['genuine_fp_rate']:.1%})",
        f"{'yes' if wa_e['attack_flagged'] else 'no'} ({wa_e['genuine_fp_rate']:.1%})",
        f"{'yes' if yt_z['attack_flagged'] else 'no'} ({yt_z['genuine_fp_rate']:.1%})",
        f"{'yes' if yt_e['attack_flagged'] else 'no'} ({yt_e['genuine_fp_rate']:.1%})",
    ])
clone_table_before(
    anchor,
    header=["N hops", "WhatsApp: z-test", "WhatsApp: elevation test", "YouTube: z-test", "YouTube: elevation test"],
    rows=rows,
)

insert_body_before(
    anchor,
    "The elevation test catches the same maximal creeping attacker at N = 3 on both edge "
    "types, against N = 9 for the z-test on WhatsApp and never within N = 20 on YouTube "
    "(N = 60.5 was the analytic prediction in Section 7.10). This is the clearest on "
    "YouTube edges specifically, exactly the case where the z-test's crossover was "
    "furthest out of reach: the elevation test needs three hops where the z-test needed "
    "sixty. Genuine false-positive rate stays in a broadly comparable range to the z-test, "
    "roughly 1 to 5% across the chain lengths tested against the z-test's roughly 1 to "
    "3%, with some fluctuation at both ends from the bootstrap resampling itself rather "
    "than a systematic trend. The choice of elevated percentile matters: at the 60th "
    "percentile the crossover moves to N = 4 with a lower false-positive rate (1.8% to "
    "2.6%), at the 80th percentile it moves to N = 2 with a higher one (3.7% to 3.9%), so "
    "the bar is a real, tunable sensitivity-versus-false-positive trade-off rather than a "
    "free improvement, and 70th percentile is reported here as a reasonable middle point, "
    "not a uniquely correct choice. Two honest caveats. First, this test is built "
    "specifically against the maximal creeping attacker construction from Section 7.10, an "
    "attacker who holds every hop at exactly its own threshold; a real attacker's behavior "
    "may differ, and the test has not been evaluated against any attack pattern besides "
    "this one. Second, the elevation test and the z-test are complementary rather than a "
    "strict replacement, the z-test remains the more natural signal for a genuine one-time "
    "large jump concentrated on a single hop, which the elevation test is not designed to "
    "catch; combining both, flagging on either signal, is a natural next step this paper "
    "does not evaluate."
)

doc.save(PAPER)
print("Inserted section 7.11.")
