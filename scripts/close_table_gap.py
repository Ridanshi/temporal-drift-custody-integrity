import re
from docx import Document

PAPER = r"D:\DF paper\paper\Temporal-Drift-Custody-Integrity.docx"
doc = Document(PAPER)

pattern = re.compile(r"(Tables?) (\d+)")


def apply_shift_to_paragraph(p):
    full = p.text
    matches = list(pattern.finditer(full))
    edits = []  # (abs_start, abs_end, new_text)
    for m in matches:
        n = int(m.group(2))
        if n < 15:
            continue
        new_n = n - 1
        old_str = m.group(0)
        new_str = f"{m.group(1)} {new_n}"
        assert len(old_str) == len(new_str), (old_str, new_str)
        edits.append((m.start(), m.end(), new_str))
    if not edits:
        return 0

    # map absolute char offsets to (run_index, local_offset)
    run_bounds = []
    pos = 0
    for i, r in enumerate(p.runs):
        run_bounds.append((pos, pos + len(r.text), i))
        pos += len(r.text)

    for start, end, new_str in edits:
        # find run(s) covering [start, end)
        covering = [rb for rb in run_bounds if rb[1] > start and rb[0] < end]
        if len(covering) == 1:
            rstart, rend, ri = covering[0]
            local_start = start - rstart
            local_end = end - rstart
            r = p.runs[ri]
            r.text = r.text[:local_start] + new_str + r.text[local_end:]
        else:
            # spans multiple runs: consume proportionally, same total length
            remaining = new_str
            for rstart, rend, ri in covering:
                r = p.runs[ri]
                seg_start = max(start, rstart) - rstart
                seg_end = min(end, rend) - rstart
                seg_len = seg_end - seg_start
                replacement_piece = remaining[:seg_len]
                remaining = remaining[seg_len:]
                r.text = r.text[:seg_start] + replacement_piece + r.text[seg_end:]
    return len(edits)


total = 0
for p in doc.paragraphs:
    total += apply_shift_to_paragraph(p)

print(f"Applied {total} shifted references.")
doc.save(PAPER)
print("Saved.")
