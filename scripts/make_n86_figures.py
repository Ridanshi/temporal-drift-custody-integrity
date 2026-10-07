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


table3 = json.load(open(r"D:\DF paper\scripts\table3_n15_result.json"))
extended = json.load(open(r"D:\DF paper\scripts\extended_n15_result.json"))

methods = ["snapshot_baseline", "drift_only", "coevasion_only", "full_system"]
method_labels = ["Snapshot\nbaseline", "Drift-only\n(ablation)", "Co-evasion-only\n(ablation)", "Full\nsystem"]
tiers = ["naive", "heuristic", "adaptive"]
tier_colors = {"naive": NAIVE_C, "heuristic": HEUR_C, "adaptive": ADAPT_C}
tier_labels = {"naive": "Naive", "heuristic": "Heuristic", "adaptive": "Adaptive"}

# ---------- Figure 5: real-data eval, single panel (WhatsApp edge) ----------
fig, ax = plt.subplots(figsize=(9.2, 5.65))
x = np.arange(len(methods))
width = 0.26
for i, tier in enumerate(tiers):
    vals = [table3["whatsapp"][m][tier]["auc"] for m in methods]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, label=tier_labels[tier], facecolor=tier_colors[tier], edgecolor="none", zorder=3)
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
ax.text(len(methods) - 0.55, 0.51, "chance", fontsize=8, color=MUTED, ha="right")
ax.set_xlim(-0.6, len(methods) - 1 + 0.6)
ax.set_ylim(0, 1.12)
ax.set_xticks(x)
ax.set_xticklabels(method_labels, fontsize=9.5)
ax.set_ylabel("AUC")
ax.set_title("Real-data evaluation (VISION, native\u2192WhatsApp edge): AUC by method and attack tier", fontsize=11, color=INK)
style_ax(ax)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9.5)
fig.tight_layout()
savefig(fig, "fig5_real_data_n86")

# ---------- Figure 9: learned baselines, 2-panel (WA | YT), IF vs OCSVM per tier ----------
ml = extended["ml_baseline_comparison"]
fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharey=True)
x = np.arange(len(tiers))
width = 0.32
for ax, edge, title in zip(axes, ["whatsapp", "youtube"], ["WhatsApp edge", "YouTube edge"]):
    if_vals = [ml[edge][t]["isolation_forest"]["auc"] for t in tiers]
    oc_vals = [ml[edge][t]["one_class_svm"]["auc"] for t in tiers]
    b1 = ax.bar(x - width / 2, if_vals, width, label="Isolation Forest", facecolor=BLUE, edgecolor="none", zorder=3)
    b2 = ax.bar(x + width / 2, oc_vals, width, label="One-Class SVM", facecolor=ORANGE, edgecolor="none", zorder=3)
    for bars in (b1, b2):
        for rect in bars:
            v = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.set_xlim(-0.6, len(tiers) - 1 + 0.6)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(["Naive", "Heuristic", "Adaptive"], fontsize=9.5)
    ax.set_title(title, fontsize=10.5, color=INK)
    style_ax(ax)
axes[0].set_ylabel("AUC")
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.03), fontsize=9.5)
fig.suptitle("Learned-baseline AUC (Isolation Forest vs. One-Class SVM) by attack tier and edge type", fontsize=10.5, color=INK, y=1.04)
fig.tight_layout()
savefig(fig, "fig9_learned_baseline_n86")

# ---------- Figure 10: ECE, snapshot baseline vs. full system, WA/YT ----------
cal = extended["calibration"]
fig, ax = plt.subplots(figsize=(6.6, 5.05))
edges = ["WhatsApp", "YouTube"]
x = np.arange(len(edges))
width = 0.32
base_vals = [cal["whatsapp"]["snapshot_baseline_ece"], cal["youtube"]["snapshot_baseline_ece"]]
full_vals = [cal["whatsapp"]["full_system_ece"], cal["youtube"]["full_system_ece"]]
b1 = ax.bar(x - width / 2, base_vals, width, label="Snapshot baseline", facecolor=BLUE, edgecolor="none", zorder=3)
b2 = ax.bar(x + width / 2, full_vals, width, label="Full system", facecolor=ORANGE, edgecolor="none", zorder=3)
for bars in (b1, b2):
    for rect in bars:
        v = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.set_xlim(-0.6, len(edges) - 1 + 0.6)
ax.set_ylim(0, max(base_vals) * 1.3)
ax.set_xticks(x)
ax.set_xticklabels(edges, fontsize=10)
ax.set_ylabel("Expected Calibration Error (lower is better)")
ax.set_title("Expected Calibration Error, adaptive tier", fontsize=11, color=INK)
style_ax(ax)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9.5)
fig.tight_layout()
savefig(fig, "fig10_ece_n86")

# ---------- Figure 11: threshold sensitivity sweep, line chart ----------
ts = extended["threshold_sensitivity"]
fig, ax = plt.subplots(figsize=(8.35, 4.8))
percentiles = [80.0, 85.0, 90.0, 95.0, 99.0]
wa_vals = [ts["whatsapp"][str(p)] for p in percentiles]
yt_vals = [ts["youtube"][str(p)] for p in percentiles]
ax.plot(percentiles, wa_vals, marker="o", color=BLUE, label="WhatsApp edge", zorder=3)
ax.plot(percentiles, yt_vals, marker="o", color=ORANGE, label="YouTube edge", zorder=3)
for px, py in zip(percentiles, wa_vals):
    ax.text(px, py + 0.01, f"{py:.3f}", ha="center", va="bottom", fontsize=7.5, color=BLUE)
for px, py in zip(percentiles, yt_vals):
    ax.text(px, py - 0.018, f"{py:.3f}", ha="center", va="top", fontsize=7.5, color=ORANGE)
ax.set_xlim(78, 101)
ax.set_ylim(0.85, 1.0)
ax.set_xticks(percentiles)
ax.set_xlabel("Calibration percentile")
ax.set_ylabel("Full-system adaptive-tier AUC")
ax.set_title("Full-system adaptive-tier AUC across calibration percentiles 80 to 99", fontsize=10.5, color=INK)
style_ax(ax)
ax.legend(frameon=False, loc="lower left", fontsize=9.5)
fig.tight_layout()
savefig(fig, "fig11_threshold_sweep_n86")

# ---------- Figure 12: cross-transform, 2-bar ----------
ct = extended["cross_transform_generalization"]
fig, ax = plt.subplots(figsize=(5.5, 4.4))
configs = ["Matched\n(fit YouTube,\nscore YouTube)", "Cross\n(fit WhatsApp,\nscore YouTube)"]
vals = [ct["matched_fit_on_youtube_auc"], ct["cross_fit_whatsapp_scoring_youtube_auc"]]
bars = ax.bar([0, 1], vals, width=0.5, color=[BLUE, ORANGE], edgecolor="none", zorder=3)
for rect, v in zip(bars, vals):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK2)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 1.08)
ax.set_xticks([0, 1])
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel("Adaptive-tier AUC (YouTube data)")
ax.set_title("Cross-transform generalization test\n(N = 86 genuine deltas per transform type)", fontsize=10.5, color=INK)
style_ax(ax)
fig.tight_layout()
savefig(fig, "fig12_cross_transform_n86")

print("ALL DONE")
