import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

GREEN = "#30B030"
RED = "#D75858"
YELLOW = "#FABD3B"
GRAY = "#CCCBC1"

COLOR_MAP = {"Yes": GREEN, "Partial": YELLOW, "No": RED, "N/A": GRAY}

systems = ["Xiang & Horv\u00e1th [1]", "C2PA [2]", "ForenGRAF-AI [5]", "Reha et al. [4]", "Mukherjee et al. [3]", "This work"]
columns = ["Temporal", "Auditable", "Calibrated", "Interpretable", "Adversarial-\naware"]

# (display_text, color_key) per system per column
data = [
    [("No", "No"), ("Partial", "Partial"), ("No", "No"), ("Yes", "Yes"), ("No", "No")],
    [("Partial\n(chain)", "Partial"), ("Yes\n(crypto)", "Yes"), ("N/A", "N/A"), ("Yes", "Yes"), ("No", "No")],
    [("Yes", "Yes"), ("Yes", "Yes"), ("Partial", "Partial"), ("Partial\n(GNN)", "Partial"), ("No", "No")],
    [("Yes", "Yes"), ("Partial", "Partial"), ("No", "No"), ("Partial", "Partial"), ("No", "No")],
    [("Partial", "Partial"), ("No", "No"), ("No", "No"), ("No", "No"), ("Yes", "Yes")],
    [("Yes", "Yes"), ("Yes", "Yes"), ("Yes", "Yes"), ("Yes", "Yes"), ("Yes", "Yes")],
]

n_rows = len(systems)
n_cols = len(columns)

fig_w, fig_h = 12.83, 7.715
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

left_margin = 0.20
right_margin = 0.99
top_header = 0.86
row_top = 0.80
row_height = 0.105
row_gap = 0.012
col_gap = 0.012

col_width = (right_margin - left_margin - col_gap * (n_cols - 1)) / n_cols

fig.text(0.02, 0.965, "Positioning against related work across five design properties (Table 22)",
          fontsize=15, ha="left", va="top", color="#111111")

for c, label in enumerate(columns):
    x = left_margin + c * (col_width + col_gap) + col_width / 2
    ax.text(x, top_header, label, fontsize=11.5, fontweight="bold", ha="center", va="bottom", color="#111111")

for r, sysname in enumerate(systems):
    y_top = row_top - r * (row_height + row_gap)
    y_center = y_top - row_height / 2
    fontweight = "bold" if sysname == "This work" else "normal"
    ax.text(left_margin - 0.015, y_center, sysname, fontsize=12, ha="right", va="center",
            fontweight=fontweight, color="#111111")

    for c in range(n_cols):
        x_left = left_margin + c * (col_width + col_gap)
        text, key = data[r][c]
        color = COLOR_MAP[key]
        text_color = "white" if key != "N/A" else "#333333"
        box = FancyBboxPatch(
            (x_left, y_top - row_height), col_width, row_height,
            boxstyle="round,pad=0,rounding_size=0.012",
            linewidth=0, facecolor=color, edgecolor="none",
        )
        ax.add_patch(box)
        ax.text(x_left + col_width / 2, y_center, text, fontsize=11.5, fontweight="bold",
                 ha="center", va="center", color=text_color)

# black outline box around the "This work" row
last_row_top = row_top - (n_rows - 1) * (row_height + row_gap)
outline = Rectangle(
    (left_margin - 0.008, last_row_top - row_height - 0.008),
    (right_margin - left_margin) + 0.016, row_height + 0.016,
    linewidth=2, edgecolor="#222222", facecolor="none",
)
ax.add_patch(outline)

# legend
legend_y = last_row_top - row_height - 0.09
legend_items = [("Yes", GREEN), ("Partial", YELLOW), ("No", RED), ("N/A", GRAY)]
legend_x_start = 0.22
swatch_w, swatch_h = 0.045, 0.035
gap = 0.16
for i, (label, color) in enumerate(legend_items):
    x0 = legend_x_start + i * gap
    swatch = FancyBboxPatch(
        (x0, legend_y - swatch_h / 2), swatch_w, swatch_h,
        boxstyle="round,pad=0,rounding_size=0.008",
        linewidth=0, facecolor=color, edgecolor="none",
    )
    ax.add_patch(swatch)
    ax.text(x0 + swatch_w + 0.012, legend_y, label, fontsize=11.5, va="center", ha="left", color="#333333")

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(r"D:\DF paper\scripts\fig13_new.png", dpi=200, facecolor="white")
print("saved", )
from PIL import Image
print(Image.open(r"D:\DF paper\scripts\fig13_new.png").size)
