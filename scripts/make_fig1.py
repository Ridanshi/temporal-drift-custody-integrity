import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

labels = ["Signal\nExtractor", "Conflict\nScorer", "Custody Graph\nBuilder", "Drift Scorer\n(per edge)", "Graph\nPropagator", "Co-Evasion\nTester", "Aggregator /\nReport"]
cols = ["#2B6CB0", "#2B6CB0", "#444444", "#E8672C", "#E8672C", "#D63638", "#1AAE78"]
fig, ax = plt.subplots(figsize=(16, 5.5), dpi=200)
ax.set_xlim(0, 16); ax.set_ylim(0, 5.5); ax.axis("off")
bw, bh, gap = 1.52, 2.1, 0.85
x0 = (16 - (7 * bw + 6 * gap)) / 2
y0 = 1.5
for i, (t, c) in enumerate(zip(labels, cols)):
    x = x0 + i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=c, lw=3))
    ax.text(x + bw / 2, y0 + bh / 2, t, ha="center", va="center", fontsize=11, color="#1B1B2F", linespacing=1.25)
    if i < 6:
        ax.add_patch(FancyArrowPatch((x + bw + 0.04, y0 + bh / 2), (x + bw + gap - 0.04, y0 + bh / 2), arrowstyle="-|>,head_length=0.55,head_width=0.28", mutation_scale=20, lw=5, color="#1B1B2F", shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt"))
ax.text(8, 4.75, "Custody-graph anomaly-detection pipeline", ha="center", va="center", fontsize=19, fontweight="bold", color="#111111")
fig.savefig("paper/new_figs/fig1_pipeline.png", facecolor="white", bbox_inches=None)
print("ok")
