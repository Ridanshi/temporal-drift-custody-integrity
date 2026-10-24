import json, sys, os
from concurrent.futures import ThreadPoolExecutor
import av, numpy as np
from custody.forge import render_intermediate, encode, wa_target_dims
from custody.features_v2 import extract_features_v2, FEATURE_NAMES as FN
from custody.vision_loader import discover_triplets

SCR = "C:/Users/Ridan/AppData/Local/Temp/claude/E--DF-paper/4ac6b769-ad6d-461e-b926-3277c9e8c3af/scratchpad/cal"
os.makedirs(SCR, exist_ok=True)
T = discover_triplets("E:/DF paper/data/vision")
F = np.load("data/vision/.feature_cache_v2.npz", allow_pickle=True)["features"].item()
rng = np.random.default_rng(0)
idx = sorted(rng.choice(len(T), 24, replace=False))
CFG = {
    "c20vf": dict(crf=20, preset="veryfast"), "c21vf": dict(crf=21, preset="veryfast"), "c22vf": dict(crf=22, preset="veryfast"),
    "c21fast": dict(crf=21, preset="fast"), "c21faster": dict(crf=21, preset="faster"), "c21uf": dict(crf=21, preset="ultrafast"),
    "c21vf_tune": dict(crf=21, preset="veryfast", deblock=(1, 1)),
}

def job(i):
    t = T[i]
    c = av.open(t["native"]); s = c.streams.video[0].codec_context; w, h = s.width, s.height; c.close()
    inter = f"{SCR}/i{i}.mkv"
    render_intermediate(t["native"], inter, None, 0, target=wa_target_dims(w, h))
    res = {}
    for k, kw in CFG.items():
        out = f"{SCR}/o{i}_{k}.mp4"
        encode(inter, out, **kw)
        res[k] = extract_features_v2(out)
    return i, res

with ThreadPoolExecutor(3) as ex:
    out = dict(ex.map(job, idx))
real = np.array([F[T[i]["whatsapp"]] for i in idx])
rs = np.array([F[T[i]["whatsapp"]] for i in range(len(T))]).std(0) + 1e-9
print("%-10s " % "cfg" + " ".join("%7s" % n[:7] for n in FN) + "   meanAbsZ")
for k in CFG:
    sim = np.array([out[i][k] for i in idx])
    bias = (sim - real).mean(0) / rs
    print("%-10s " % k + " ".join("%7.2f" % b for b in bias) + "   %.3f" % np.abs((sim - real) / rs).mean())

print("std ratio sim/real:")
for k in CFG:
    sim = np.array([out[i][k] for i in idx])
    print("%-10s " % k + " ".join("%7.2f" % r for r in sim.std(0) / (real.std(0) + 1e-9)))
