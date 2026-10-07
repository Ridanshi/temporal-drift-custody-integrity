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
BLUE = "#2a78d6"
ORANGE = "#eb6834"

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


def style_ax(ax):
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASE)
    ax.spines["bottom"].set_color(BASE)


def savefig(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)


table3 = json.load(open(r"D:\DF paper\scripts\table3_n86_fixed.json"))
dfd = json.load(open(r"D:\DF paper\scripts\dfd_eval_n86_fixed.json"))
celebdf = json.load(open(r"D:\DF paper\scripts\celebdf_eval_n86_fixed.json"))

methods = ["snapshot_baseline", "drift_only", "coevasion_only", "full_system"]
method_labels = ["Snapshot\nbaseline", "Drift-only\n(ablation)", "Co-evasion-only\n(ablation)", "Full\nsystem"]


def make_compare_fig(synthetic_tier_key, real_data, real_label, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharey=True)
    x = np.arange(len(methods))
    width = 0.32
    for ax, edge, edge_title in zip(axes, ["whatsapp", "youtube"], ["WhatsApp edge", "YouTube edge"]):
        synth_vals = [table3[edge][m][synthetic_tier_key]["auc"] for m in methods]
        real_vals = [real_data[edge][m]["auc"] for m in methods]
        b1 = ax.bar(x - width / 2, synth_vals, width, label="Synthetic (Table 3)", facecolor=BLUE, edgecolor="none", zorder=3)
        b2 = ax.bar(x + width / 2, real_vals, width, label=real_label, facecolor=ORANGE, edgecolor="none", zorder=3)
        for bars in (b1, b2):
            for rect in bars:
                v = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.8, color=INK2)
        ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.set_xlim(-0.6, len(methods) - 1 + 0.6)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=8.6)
        ax.set_title(edge_title, fontsize=10.5, color=INK)
        style_ax(ax)
    axes[0].set_ylabel("AUC")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.03), fontsize=9.5)
    fig.suptitle(title, fontsize=10.2, color=INK, y=1.04)
    fig.tight_layout()
    savefig(fig, fname)


make_compare_fig(
    "naive", dfd, "Real DFD naive",
    "Naive-tier AUC: synthetic construction (Table 3) vs. real DFD deepfake tamper (Table 7)",
    "fig7_dfd_naive_fixed",
)

make_compare_fig(
    "heuristic", celebdf, "Real Celeb-DF heuristic",
    "Heuristic-tier AUC: synthetic construction (Table 3) vs. real Celeb-DF deepfake + off-the-shelf re-encode (Table 8)",
    "fig8_celebdf_heuristic_fixed",
)

print("ALL DONE")
