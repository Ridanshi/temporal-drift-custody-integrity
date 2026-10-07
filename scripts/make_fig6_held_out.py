import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"D:\DF paper\scripts"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#ffffff"
NAIVE_C = "#2a78d6"
HEUR_C = "#eb6834"
ADAPT_C = "#3fa66b"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": INK,
        "axes.edgecolor": BASE,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)

R = json.load(open(r"D:\DF paper\scripts\held_out_eval_n15_result.json"))

methods = ["snapshot_baseline", "drift_only", "coevasion_only", "full_system"]
method_labels = ["Snapshot\nbaseline", "Drift-only\n(ablation)", "Co-evasion-only\n(ablation)", "Full\nsystem"]
tiers = ["naive", "heuristic", "adaptive"]
tier_colors = {"naive": NAIVE_C, "heuristic": HEUR_C, "adaptive": ADAPT_C}
tier_labels = {"naive": "Naive", "heuristic": "Heuristic", "adaptive": "Adaptive"}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
x = np.arange(len(methods))
width = 0.26

for ax, edge, title in zip(axes, ["whatsapp", "youtube"], ["WhatsApp edge", "YouTube edge"]):
    for i, tier in enumerate(tiers):
        means = []
        stds = []
        for m in methods:
            aucs = np.array(R[edge][m][tier]["auc_per_fold"])
            means.append(aucs.mean())
            stds.append(aucs.std())
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=tier_labels[tier],
            facecolor=tier_colors[tier],
            edgecolor="none",
            zorder=3,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capthick": 1},
        )
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.set_xlim(-0.6, len(methods) - 1 + 0.6)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=8.6)
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASE)
    ax.spines["bottom"].set_color(BASE)

axes[0].set_ylabel("AUC")
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.03), fontsize=9)
fig.suptitle(
    "Leave-one-device-out held-out validation, AUC by method and attack tier (N = 15 folds, error bars = \u00b1 1 std)",
    fontsize=10.2,
    color=INK,
    y=1.04,
)
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_held_out_n15.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print("wrote fig6_held_out_n15.png")
