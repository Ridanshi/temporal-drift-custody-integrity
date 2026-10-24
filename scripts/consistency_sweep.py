import re
from docx import Document

PAPER = r"D:\DF paper\paper\Temporal-Drift-Custody-Integrity.docx"
doc = Document(PAPER)

full_text = []
table_captions = []
figure_captions = []
section_headings = []

for p in doc.paragraphs:
    t = p.text.strip()
    if not t:
        continue
    full_text.append(t)
    m = re.match(r"^Table (\d+)\.", t)
    if m:
        table_captions.append((int(m.group(1)), t[:90]))
    m = re.match(r"^Figure (\d+)\.", t)
    if m:
        figure_captions.append((int(m.group(1)), t[:90]))
    m = re.match(r"^(\d+(?:\.\d+)*)\s+[A-Z]", t)
    if m and len(t) < 120:
        section_headings.append((m.group(1), t))

joined = "\n".join(full_text)

print("=== TABLE CAPTIONS (document order) ===")
expected = 1
issues = []
for n, txt in table_captions:
    flag = "" if n == expected else f"  <-- EXPECTED {expected}"
    if flag:
        issues.append(f"Table caption out of sequence: got {n}, expected {expected} ({txt})")
    print(f"{n}: {txt}{flag}")
    expected = n + 1

print()
print("=== FIGURE CAPTIONS (document order) ===")
expected = 1
for n, txt in figure_captions:
    flag = "" if n == expected else f"  <-- EXPECTED {expected}"
    if flag:
        issues.append(f"Figure caption out of sequence: got {n}, expected {expected} ({txt})")
    print(f"{n}: {txt}{flag}")
    expected = n + 1

print()
print("=== SECTION HEADINGS (document order, dotted-numeric only) ===")
for num, txt in section_headings:
    print(f"{num}: {txt}")

max_table = max((n for n, _ in table_captions), default=0)
max_figure = max((n for n, _ in figure_captions), default=0)

print()
print("=== IN-TEXT TABLE REFERENCES ===")
table_ref_nums = set()
for m in re.finditer(r"Tables? (\d+)(?:\s*(?:through|to|and|,|-)\s*(\d+))?", joined):
    a = int(m.group(1))
    table_ref_nums.add(a)
    if m.group(2):
        b = int(m.group(2))
        table_ref_nums.add(b)
bad_table_refs = sorted(n for n in table_ref_nums if n > max_table or n < 1)
print(f"Referenced table numbers: {sorted(table_ref_nums)}")
print(f"Max table caption number: {max_table}")
if bad_table_refs:
    issues.append(f"In-text references to nonexistent table numbers: {bad_table_refs}")

print()
print("=== IN-TEXT FIGURE REFERENCES ===")
figure_ref_nums = set()
for m in re.finditer(r"Figures? (\d+)(?:\s*(?:through|to|and|,|-)\s*(\d+))?", joined):
    a = int(m.group(1))
    figure_ref_nums.add(a)
    if m.group(2):
        b = int(m.group(2))
        figure_ref_nums.add(b)
bad_figure_refs = sorted(n for n in figure_ref_nums if n > max_figure or n < 1)
print(f"Referenced figure numbers: {sorted(figure_ref_nums)}")
print(f"Max figure caption number: {max_figure}")
if bad_figure_refs:
    issues.append(f"In-text references to nonexistent figure numbers: {bad_figure_refs}")

# check for duplicate table/figure numbers
from collections import Counter
tdup = [n for n, c in Counter(n for n, _ in table_captions).items() if c > 1]
fdup = [n for n, c in Counter(n for n, _ in figure_captions).items() if c > 1]
if tdup:
    issues.append(f"Duplicate table caption numbers: {tdup}")
if fdup:
    issues.append(f"Duplicate figure caption numbers: {fdup}")

print()
print("=== SECTION NUMBER GAPS/DUPES (7.x range) ===")
sevens = [num for num, _ in section_headings if num.startswith("7.") and num.count(".") == 1]
seven_nums = sorted(set(int(s.split(".")[1]) for s in sevens), key=lambda x: x)
seq_issue = []
for i in range(1, max(seven_nums, default=0) + 1):
    if i not in seven_nums:
        seq_issue.append(i)
print(f"7.x subsections present: {seven_nums}")
if seq_issue:
    issues.append(f"Missing 7.x subsection numbers: {seq_issue}")

print()
print("=== SUMMARY ===")
if issues:
    for i in issues:
        print("ISSUE:", i)
else:
    print("No issues found.")
