import os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from custody.local_features import decode_small, local_pair_features
from custody.vision_loader import discover_triplets
sys.path.insert(0, "scripts")
from eval_v2 import forged_path, OPS

T = discover_triplets("E:/DF paper/data/vision")
import sys as _s; _s.path.insert(0, 'scripts'); import srcshim; srcshim.apply(T)
CACHE = os.environ.get("LOCC", "work/local_features.npz")
TIERS = os.environ.get("TIERS", "simctl,platform,naive,adaptive,adaptive_drift_only,adaptive_fusion,platform_x,s25,s50").split(",")
EXTRA = ("text", "inpaint", "cut")
D = {}

def targets(i):
    t = T[i]; out = [t["whatsapp"]]
    for tier in TIERS:
        for op in ([None] if tier == "simctl" else (EXTRA if tier == "platform_x" else OPS)):
            p = forged_path(tier, i, op)
            if os.path.exists(p): out.append(p)
    return [p for p in out if (t["native"], p) not in D]

def job(i):
    tg = targets(i)
    if not tg: return {}
    res = {}
    try:
        src = decode_small(T[i]["native"])
        for p in tg:
            try: res[(T[i]["native"], p)] = local_pair_features(src, decode_small(p))
            except Exception as e: print("skip", p[-40:], repr(e)[:60], flush=True)
    except Exception as e:
        print("clip fail", i, repr(e)[:60], flush=True)
    return res

if __name__ == "__main__":
    if os.path.exists(CACHE): D = np.load(CACHE, allow_pickle=True)["features"].item()
    t0 = time.time()
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 2))) as ex:
        for n, r in enumerate(ex.map(job, range(len(T)))):
            D.update(r)
            if n % 4 == 3:
                np.savez(CACHE, features=np.array(D, dtype=object)); print(n + 1, len(D), round(time.time() - t0), "s", flush=True)
    np.savez(CACHE, features=np.array(D, dtype=object)); print("finished", len(D), round(time.time() - t0), "s")
