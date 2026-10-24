import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
sys.path.insert(0, "scripts")
import eval_v2 as E

OUT = "paper/new_figs"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 200})
COL = {"snapshot_conflict": "#444444", "mismatch_only": "#8c6bb1", "drift_only": "#e6550d", "coevasion_only": "#3182bd",
       "full_system": "#31a354", "isolation_forest": "#bdbdbd", "ocsvm": "#969696"}
LAB = {"snapshot_conflict": "Snapshot conflict", "mismatch_only": "Mismatch-only", "drift_only": "Drift-only",
       "coevasion_only": "Co-evasion-only", "full_system": "Full system", "isolation_forest": "Isolation Forest", "ocsvm": "One-Class SVM"}
forged = E.load_forged(); tiers = [t for t in ("naive", "platform", "adaptive", "adaptive_drift_only", "s25", "s50") if any(k[0] == t for k in forged)]
dets = E.fold_detectors(); sc = E.collect(forged, tiers, dets)
M = E.METHODS
TN = {"naive": "Naive\n(stock re-encode)", "platform": "Platform-aware\n(public spec)", "adaptive": "Adaptive white-box\n(vs full system)", "adaptive_drift_only": "Adaptive white-box\n(vs drift model)"}

def pg(t): return [(t, op) for op in E.OPS if (t, op) in sc]

# Fig A: grouped AUC bars
main = [t for t in ("naive", "platform", "adaptive", "adaptive_drift_only") if t in tiers]
fig, ax = plt.subplots(figsize=(7.4, 3.3)); w = 0.8 / len(M)
res = {}
for j, m in enumerate(M):
    vals = [E.auc_ci(sc, pg(t), m, n_boot=500) for t in main]; res[m] = vals
    x = np.arange(len(main)) + j * w - 0.4 + w / 2
    ax.bar(x, [v[0] for v in vals], w, color=COL[m], label=LAB[m], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], error_kw=dict(lw=0.7, capsize=1.5))
ax.axhline(0.5, color="k", ls="--", lw=0.8); ax.set_xticks(range(len(main))); ax.set_xticklabels([TN[t] for t in main]); ax.set_ylabel("AUC (genuine vs forged)"); ax.set_ylim(0, 1.05)
ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/figA_auc.png"); plt.close(fig)

# Fig B: ROC platform-aware and adaptive
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
for ax, t in zip(axs, ("platform", "adaptive")):
    g = {m: E.pooled(sc, "genuine", m) for m in M}
    for m in ("snapshot_conflict", "drift_only", "coevasion_only", "full_system"):
        p = np.concatenate([E.pooled(sc, x, m) for x in pg(t)])
        f, tt, _ = roc_curve(np.r_[np.zeros(len(g[m])), np.ones(len(p))], np.r_[g[m], p]); ax.plot(f, tt, color=COL[m], label=LAB[m], lw=1.6)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False-positive rate"); ax.set_title(TN[t].replace("\n", " "), fontsize=9)
axs[0].set_ylabel("True-positive rate"); axs[1].legend(fontsize=7, loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/figB_roc.png"); plt.close(fig)

# Fig C: per-edit heatmap (platform + adaptive)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
ms = ["snapshot_conflict", "drift_only", "coevasion_only", "full_system", "ocsvm"]
for ax, t in zip(axs, ("platform", "adaptive")):
    A = np.array([[E.auc_ci(sc, [(t, op)], m, n_boot=2)[0] for m in ms] for op in E.OPS])
    im = ax.imshow(A, vmin=0.2, vmax=1.0, cmap="RdYlGn", aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]): ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([LAB[m].replace(" ", "\n") for m in ms], fontsize=7); ax.set_title(TN[t].replace("\n", " "), fontsize=9); ax.grid(False)
axs[0].set_yticks(range(len(E.OPS))); axs[0].set_yticklabels(["copy-move", "blur", "splice", "recolor", "frame freeze"])
fig.tight_layout(); fig.savefig(f"{OUT}/figC_edits.png"); plt.close(fig)

# Fig D: chain AUCs
ca = json.load(open("scripts/eval_chains_auc.json")); names = {"max_edge_drift": "Per-edge drift (max)", "path_z": "Path z-test", "e2e_drift": "End-to-end drift", "elev_count": "Elevated-hop count"}
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
for ax, scen, ttl in zip(axs, ("tamper2", "creep"), ("One tampered hop", "Creeping tamper (three hops)")):
    w = 0.2
    for j, (k, lab) in enumerate(names.items()):
        ax.bar(np.arange(3) + j * w - 0.3, [ca[f"{scen}_{m}"][k] for m in (2, 3, 4)], w, label=lab, color=["#e6550d", "#756bb1", "#3182bd", "#bdbdbd"][j])
    ax.axhline(0.5, color="k", ls="--", lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels(["2 hops", "3 hops", "4 hops"]); ax.set_title(ttl, fontsize=9); ax.set_ylim(0.4, 1.0)
axs[0].set_ylabel("AUC (genuine vs tampered chains)"); axs[0].legend(fontsize=6.5, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/figD_chains.png"); plt.close(fig)

# Fig E: strength sweep
if "s25" in tiers:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ev = {}
    for m in ("snapshot_conflict", "drift_only", "full_system", "ocsvm"):
        ys = []
        for t in ("s25", "s50", "platform"):
            sub = {d: sc["genuine"][m][d] for d in E.DEVS}
            ys.append(E.auc_ci(sc, pg(t), m, n_boot=2)[0])
        ev[m] = ys; ax.plot([0.25, 0.5, 1.0], ys, "o-", color=COL[m], label=LAB[m])
    ax.axhline(0.5, color="k", ls="--", lw=0.8); ax.set_xlabel("Edit strength"); ax.set_ylabel("AUC"); ax.set_xticks([0.25, 0.5, 1.0]); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/figE_strength.png"); plt.close(fig)
    json.dump(ev, open("scripts/strength_auc.json", "w"), indent=1)

# Fig F: adaptive search curve
meta = json.load(open("work/adaptive_meta.json"))
tr = np.array([m["trace"][:11] for m in meta.values() if len(m["trace"]) >= 11])
best = np.minimum.accumulate(tr, axis=1)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8))
axs[0].plot(range(1, 12), best.mean(0), "o-", color=COL["full_system"]); axs[0].set_xlabel("Encoder trials by the attacker"); axs[0].set_ylabel("Mean full-system score of best forgery")
axs[1].plot(range(1, 12), (best > 0).mean(0), "o-", color=COL["drift_only"]); axs[1].set_xlabel("Encoder trials by the attacker"); axs[1].set_ylabel("Fraction of forgeries still flagged"); axs[1].set_ylim(0, 1)
fig.tight_layout(); fig.savefig(f"{OUT}/figF_search.png"); plt.close(fig)
json.dump(dict(mean_best=best.mean(0).tolist(), flagged=(best > 0).mean(0).tolist(), n=len(tr)), open("scripts/search_curve.json", "w"), indent=1)
json.dump({m: [list(v) for v in res[m]] for m in M}, open("scripts/main_auc_ci.json", "w"), indent=1)
print("figs done", os.listdir(OUT))
