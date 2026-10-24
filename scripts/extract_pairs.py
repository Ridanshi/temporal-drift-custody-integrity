"""Pair (edge) features for genuine WA/YT edges and all forged files whose source is the native clip."""
import os, sys, time, glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from custody.features_v2 import pair_features_multi
from custody.vision_loader import discover_triplets
sys.path.insert(0, "scripts")
from eval_v2 import forged_path, OPS

T = discover_triplets("E:/DF paper/data/vision")
import sys as _s; _s.path.insert(0, 'scripts'); import srcshim; srcshim.apply(T)
CACHE = os.environ.get("PAIRC", "work/pair_features.npz")
TIERS = os.environ.get("TIERS", "simctl,platform,naive,adaptive").split(",")


def clip_job(i):
    t = T[i]
    tg = [t["whatsapp"], t["youtube"]]
    for tier in TIERS:
        for op in ([None] if tier == "simctl" else (("text", "inpaint", "cut") if tier == "platform_x" else OPS)):
            p = forged_path(tier, i, op)
            if os.path.exists(p):
                tg.append(p)
    done = D
    tg = [p for p in tg if (t["native"], p) not in done]
    if not tg:
        return i, {}
    vecs = pair_features_multi(t["native"], tg)
    return i, {(t["native"], p): v for p, v in zip(tg, vecs)}


if __name__ == "__main__":
    D = {k: v for k, v in np.load(CACHE, allow_pickle=True)["features"].item().items()} if os.path.exists(CACHE) else {}
    t0 = time.time()
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 3))) as ex:
        for n, (i, d) in enumerate(ex.map(clip_job, range(len(T)))):
            D.update(d)
            if n % 10 == 9:
                np.savez(CACHE, features=np.array(D, dtype=object)); print(n + 1, round(time.time() - t0), "s", flush=True)
    np.savez(CACHE, features=np.array(D, dtype=object)); print("finished", len(D), round(time.time() - t0), "s")
