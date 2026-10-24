"""Header-only forgery: stock-encoded forgeries whose declared fields are rewritten to the genuine WhatsApp values."""
import json, sys
import numpy as np
sys.path.insert(0, "scripts")
import eval_v2 as E

forged = E.load_forged(("naive", "platform"))
variants = {}
for name in ("honest_declared", "copy_clip_header", "median_header"):
    f2 = {}
    for k, v in forged.items():
        if k[0] != "naive": continue
        v = v.copy()
        if name == "copy_clip_header": v[:4] = E.W[k[1]][:4]
        elif name == "median_header": v[:4] = np.median(E.W[:, :4], axis=0)
        f2[("naive", k[1], k[2])] = v
    variants[name] = f2
dets = E.fold_detectors()
out = {}
for name, f2 in variants.items():
    sc = E.collect(f2, ["naive"], dets)
    out[name] = {}
    for m in E.METHODS:
        a, lo, hi = E.auc_ci(sc, [("naive", op) for op in E.OPS], m, n_boot=500)
        out[name][m] = dict(auc=a, lo=lo, hi=hi, tpr10=E.tpr_at_fpr(sc, [("naive", op) for op in E.OPS], m))
    print(name, {m[:6]: round(x["auc"], 2) for m, x in out[name].items()})
json.dump(out, open("scripts/header_forgery.json", "w"), indent=1)
