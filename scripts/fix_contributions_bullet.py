from docx import Document

PAPER = r"D:\DF paper\paper\Temporal-Drift-Custody-Integrity.docx"
doc = Document(PAPER)


def replace_span(p, old, new):
    full = p.text
    assert full.count(old) == 1, f"expected exactly 1 match, got {full.count(old)}: {old!r}"
    idx = full.index(old)
    end_idx = idx + len(old)

    pos = 0
    start_run = end_run = None
    for i, r in enumerate(p.runs):
        run_start = pos
        run_end = pos + len(r.text)
        if start_run is None and run_start <= idx < run_end:
            start_run = i
            start_local = idx - run_start
        if run_start < end_idx <= run_end:
            end_run = i
            end_local = end_idx - run_start
        pos = run_end
    assert start_run is not None and end_run is not None

    if start_run == end_run:
        r = p.runs[start_run]
        r.text = r.text[:start_local] + new + r.text[end_local:]
    else:
        p.runs[start_run].text = p.runs[start_run].text[:start_local] + new
        for k in range(start_run + 1, end_run):
            p.runs[k].text = ""
        p.runs[end_run].text = p.runs[end_run].text[end_local:]


replace_span(
    doc.paragraphs[9],
    "(4) A three-tier adversarial evaluation protocol validated end-to-end on synthetic "
    "custody chains and, on an initial 8-triplet real-video subset, a statistically "
    "significant improvement (McNemar's test, p < 10\u207b\u2075 on every tier and edge type) "
    "over a reimplemented snapshot-fusion baseline.",
    "(4) A three-tier adversarial evaluation protocol validated end-to-end on synthetic "
    "custody chains and, on an 86-triplet real-video subset across 15 devices, a "
    "statistically significant improvement (McNemar's test, p < 0.005 throughout) over "
    "a reimplemented snapshot-fusion baseline on the heuristic and adaptive tiers, the "
    "tiers the co-evasion test is designed to target, though not yet on the naive tier "
    "at this sample size, where both methods are already near-ceiling.",
)

doc.save(PAPER)
print("Updated Contributions bullet (4) to cite the N=86 result.")
