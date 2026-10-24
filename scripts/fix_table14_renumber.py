from docx import Document

PAPER = r"D:\DF paper\paper\Temporal-Drift-Custody-Integrity.docx"
doc = Document(PAPER)


def replace_in_paragraph(p, old, new):
    full = p.text
    assert old in full, f"substring not found: {old!r} in {full!r}"
    idx = full.index(old)
    pos = 0
    start_run = None
    for i, r in enumerate(p.runs):
        run_start = pos
        run_end = pos + len(r.text)
        if run_start <= idx < run_end:
            start_run = i
            break
        pos = run_end
    assert start_run is not None

    # find end run
    end_idx = idx + len(old)
    pos = 0
    end_run = None
    for i, r in enumerate(p.runs):
        run_start = pos
        run_end = pos + len(r.text)
        if run_start < end_idx <= run_end:
            end_run = i
            break
        pos = run_end
    assert end_run is not None

    if start_run == end_run:
        r = p.runs[start_run]
        local_start = idx - sum(len(p.runs[k].text) for k in range(start_run))
        r.text = r.text[:local_start] + new + r.text[local_start + len(old):]
    else:
        pos = 0
        local_start = idx - sum(len(p.runs[k].text) for k in range(start_run))
        local_end = end_idx - sum(len(p.runs[k].text) for k in range(end_run))
        p.runs[start_run].text = p.runs[start_run].text[:local_start] + new
        for k in range(start_run + 1, end_run):
            p.runs[k].text = ""
        p.runs[end_run].text = p.runs[end_run].text[local_end:]


targets = [
    (30, "Table 14 (Section 8.3)", "Table 23 (Section 8.4)"),
    (173, "Table 14 extends Table 1's", "Table 23 extends Table 1's"),
    (174, "Table 14.", "Table 23."),
    (176, "(Table 14", "(Table 23"),
]

for idx, old, new in targets:
    p = doc.paragraphs[idx]
    assert old in p.text, f"para {idx} missing {old!r}: {p.text!r}"
    replace_in_paragraph(p, old, new)
    print(f"para {idx} fixed -> {p.text[:150]}")

doc.save(PAPER)
print("Saved.")
