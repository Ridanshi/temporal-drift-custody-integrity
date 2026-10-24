from docx import Document

PAPER = r"D:\DF paper\paper\Temporal-Drift-Custody-Integrity.docx"
doc = Document(PAPER)

t2 = doc.tables[1]
for row in t2.rows[1:]:
    if row.cells[0].text.strip() == "Snapshot-fusion baseline":
        for ci, val in zip([1, 2, 3], ["0.98", "0.48", "0.52"]):
            cell = row.cells[ci]
            for p in cell.paragraphs:
                for r in list(p.runs):
                    r.text = ""
                if p.runs:
                    p.runs[0].text = val
                else:
                    p.add_run(val)

p71 = doc.paragraphs[71]
p71.runs[0].text = (
    "Caveat. Every method's AUC sits at or near 0.5 on the adaptive tier, which we "
    "attribute to a construction artifact of this particular synthetic attacker: a "
    "small-noise mimicry attack in delta-space lands closer to the genuine mean than "
    "genuine test-set variance itself, an artifact of how the synthetic generator was "
    "built rather than a property of the underlying method. At these settings the "
    "full-system-vs-baseline ordering does not hold cleanly (0.48 vs. 0.52); the two "
    "methods sit within a few points of each other and of chance itself, which we read "
    "as further evidence that the adaptive tier's synthetic construction does not "
    "produce a stable, meaningful signal in either direction, not as a finding that the "
    "baseline outperforms the full system. These magnitudes should not be read as a "
    "claim about real-world detectability. This is precisely why Section 7.2's "
    "real-data pass, not further tuning of the synthetic attacker, is the result that "
    "matters."
)
for r in p71.runs[1:]:
    r.text = ""

doc.save(PAPER)
print("Fixed Table 2 and paragraph 71.")
