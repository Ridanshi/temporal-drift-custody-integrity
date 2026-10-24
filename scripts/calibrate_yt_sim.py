import os, numpy as np, av
from concurrent.futures import ThreadPoolExecutor
from custody.forge import render_intermediate
from custody.features_v2 import extract_features_v2, FEATURE_NAMES as FN
from custody.vision_loader import discover_triplets
SCR = "C:/Users/Ridan/AppData/Local/Temp/claude/E--DF-paper/4ac6b769-ad6d-461e-b926-3277c9e8c3af/scratchpad/caly"
os.makedirs(SCR, exist_ok=True)
T = discover_triplets("E:/DF paper/data/vision")
F = np.load("data/vision/.feature_cache_v2.npz", allow_pickle=True)["features"].item()
rng = np.random.default_rng(1)
idx = sorted(rng.choice(len(T), 12, replace=False))
base = dict(profile="high", keyint=90, bframes=2, preset="medium", scenecut=40)
CFG = {"crf20": dict(base, crf=20), "crf23": dict(base, crf=23), "crf26": dict(base, crf=26), "crf23_k60b3": dict(base, crf=23, bframes=3, keyint=60)}
def job(i):
    t = T[i]; c = av.open(t["native"]); cc = c.streams.video[0].codec_context; w, h = cc.width, cc.height; c.close()
    res = {}
    for k, kw in CFG.items():
        o = f"{SCR}/o{i}_{k}.mp4"
        render_intermediate(t["native"], o, None, 0, target=(w, h), direct=kw)
        res[k] = extract_features_v2(o)
    return i, res
with ThreadPoolExecutor(1) as ex:
    out = dict(ex.map(job, idx))
real = np.array([F[T[i]["youtube"]] for i in idx]); rs = np.array([F[T[i]["youtube"]] for i in range(len(T))]).std(0) + 1e-9
print("%-14s " % "cfg" + " ".join("%7s" % n[:7] for n in FN) + "   meanAbsZ")
for k in CFG:
    sim = np.array([out[i][k] for i in idx])
    print("%-14s " % k + " ".join("%7.2f" % b for b in (sim - real).mean(0) / rs) + "   %.3f" % np.abs((sim - real) / rs).mean())
