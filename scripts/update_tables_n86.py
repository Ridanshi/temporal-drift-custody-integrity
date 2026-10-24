from docx import Document

PAPER = r"D:\DF paper\paper\Temporal-Drift-Custody-Integrity.docx"
doc = Document(PAPER)


def set_row_cells(row, values):
    for ci, val in enumerate(values, start=1):
        cell = row.cells[ci]
        for p in cell.paragraphs:
            for r in list(p.runs):
                r.text = ""
            if p.runs:
                p.runs[0].text = val
            else:
                p.add_run(val)


# --- Table 3 (doc.tables[2]): AUC by method/tier, real single-split, N=86 ---
t3 = doc.tables[2]
table3_data = {
    "Snapshot-fusion baseline": ["0.91 / 0.91", "0.86 / 0.86", "0.86 / 0.85"],
    "Drift-only (ablation)": ["1.00 / 1.00", "1.00 / 1.00", "0.95 / 0.94"],
    "Co-evasion-only (ablation)": ["0.80 / 0.80", "0.95 / 0.95", "0.45 / 0.45"],
    "Full system": ["1.00 / 1.00", "1.00 / 1.00", "0.90 / 0.92"],
}
for row in t3.rows[1:]:
    method = row.cells[0].text.strip()
    if method in table3_data:
        set_row_cells(row, table3_data[method])
    else:
        print("WARNING unmatched Table 3 row:", method)

# --- Table 4 (doc.tables[3]): McNemar ---
t4 = doc.tables[3]
table4_data = {
    "WhatsApp / naive": ["12", "9", "0.19", "0.663"],
    "WhatsApp / heuristic": ["32", "9", "11.80", "5.9e-04"],
    "WhatsApp / adaptive": ["27", "9", "8.03", "4.6e-03"],
    "YouTube / naive": ["16", "10", "0.96", "0.327"],
    "YouTube / heuristic": ["36", "10", "13.59", "2.3e-04"],
    "YouTube / adaptive": ["32", "10", "10.50", "1.2e-03"],
}
for row in t4.rows[1:]:
    key = row.cells[0].text.strip()
    if key in table4_data:
        set_row_cells(row, table4_data[key])
    else:
        print("WARNING unmatched Table 4 row:", key)

# --- Table 9 (doc.tables[8]): learned baselines ---
t9 = doc.tables[8]
table9_data = {
    "WhatsApp / naive": ["0.95", "0.94"],
    "WhatsApp / heuristic": ["0.77", "0.87"],
    "WhatsApp / adaptive": ["0.73", "0.87"],
    "YouTube / naive": ["0.99", "1.00"],
    "YouTube / heuristic": ["0.83", "0.85"],
    "YouTube / adaptive": ["0.63", "0.88"],
}
for row in t9.rows[1:]:
    key = row.cells[0].text.strip()
    if key in table9_data:
        set_row_cells(row, table9_data[key])
    else:
        print("WARNING unmatched Table 9 row:", key)

# --- Table 10 (doc.tables[9]): ECE ---
t10 = doc.tables[9]
table10_data = {
    "WhatsApp": ["0.269", "0.203"],
    "YouTube": ["0.276", "0.202"],
}
for row in t10.rows[1:]:
    key = row.cells[0].text.strip()
    if key in table10_data:
        set_row_cells(row, table10_data[key])
    else:
        print("WARNING unmatched Table 10 row:", key)

# --- Table 11 (doc.tables[10]): threshold sensitivity ---
t11 = doc.tables[10]
table11_data = {
    "80th": ["0.940", "0.927"],
    "85th": ["0.929", "0.933"],
    "90th (reported)": ["0.934", "0.939"],
    "95th": ["0.907", "0.942"],
    "99th": ["0.898", "0.898"],
}
for row in t11.rows[1:]:
    key = row.cells[0].text.strip()
    if key in table11_data:
        set_row_cells(row, table11_data[key])
    else:
        print("WARNING unmatched Table 11 row:", key)

# --- Table 13 (doc.tables[12]): cross-transform ---
t13 = doc.tables[12]
table13_data = {
    "Matched (fit on YouTube, score YouTube)": ["0.908"],
    "Cross (fit on WhatsApp, score YouTube)": ["0.965"],
}
for row in t13.rows[1:]:
    key = row.cells[0].text.strip()
    if key in table13_data:
        set_row_cells(row, table13_data[key])
    else:
        print("WARNING unmatched Table 13 row:", key)

# --- Table 7 (doc.tables[6]): DFD naive-tier, synthetic column + real columns ---
t7 = doc.tables[6]
table7_data = {
    # method: [synthetic_naive_auc, real_dfd_auc, real_dfd_f1, real_ci]
    "Snapshot-fusion baseline": ["0.91 / 0.91", "0.94 / 0.97", "0.79 / 0.77", "[0.89, 0.98] / [0.94, 0.99]"],
    "Drift-only (ablation)": ["1.00 / 1.00", "0.97 / 1.00", "0.80 / 0.87", "[0.94, 0.99] / [0.99, 1.00]"],
    "Co-evasion-only (ablation)": ["0.80 / 0.80", "1.00 / 1.00", "0.70 / 0.70", "[1.00, 1.00] / [1.00, 1.00]"],
    "Full system": ["1.00 / 1.00", "1.00 / 1.00", "0.80 / 0.81", "[0.99, 1.00] / [0.99, 1.00]"],
}
for row in t7.rows[1:]:
    method = row.cells[0].text.strip()
    if method in table7_data:
        set_row_cells(row, table7_data[method])
    else:
        print("WARNING unmatched Table 7 row:", method)

# --- Table 8 (doc.tables[7]): Celeb-DF heuristic-tier, synthetic column + real columns ---
t8 = doc.tables[7]
table8_data = {
    "Snapshot-fusion baseline": ["0.86 / 0.86", "1.00 / 1.00", "0.91 / 0.89", "[1.00, 1.00] / [1.00, 1.00]"],
    "Drift-only (ablation)": ["1.00 / 1.00", "1.00 / 1.00", "0.90 / 0.90", "[1.00, 1.00] / [1.00, 1.00]"],
    "Co-evasion-only (ablation)": ["0.95 / 0.95", "1.00 / 1.00", "0.75 / 0.75", "[1.00, 1.00] / [1.00, 1.00]"],
    "Full system": ["1.00 / 1.00", "1.00 / 1.00", "0.84 / 0.85", "[1.00, 1.00] / [1.00, 1.00]"],
}
for row in t8.rows[1:]:
    method = row.cells[0].text.strip()
    if method in table8_data:
        set_row_cells(row, table8_data[method])
    else:
        print("WARNING unmatched Table 8 row:", method)

doc.save(PAPER)
print("Saved all table updates.")
