"""Edge features for every hop of every generated chain, plus native->hop end-to-end features."""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from custody.features_v2 import pair_features_multi
from custody.vision_loader import discover_triplets
sys.path.insert(0, "scripts")
from gen_chains import ROUTE, OUT

T = discover_triplets("E:/DF paper/data/vision")
CACHE = "work/chain_pairs.npz"


def chain_files(scen, route, i, op):
    t = T[i]
    first = t["whatsapp"] if ROUTE[route][0] == "WA" else t["youtube"]
    tag = f"{scen}_{route}_{i:03d}_{op or 'none'}"
    return [first] + [f"{OUT}/{tag}_h{k}.mp4" for k in (2, 3, 4)]


def job(key):
    scen, route, i, op = key
    f = chain_files(scen, route, i, op)
    if not all(os.path.exists(p) for p in f):
        return key, None
    out = {}
    for k in (1, 2, 3):
        out[("hop", k + 1)] = pair_features_multi(f[k - 1], [f[k]])[0]
    e2e = pair_features_multi(T[i]["native"], f[1:])
    for k, v in zip((2, 3, 4), e2e):
        out[("e2e", k)] = v
    return key, out


if __name__ == "__main__":
    done = np.load("work/chain_done.npz", allow_pickle=True)["features"].item()
    keys = [k for k in done]
    res = np.load(CACHE, allow_pickle=True)["features"].item() if os.path.exists(CACHE) else {}
    todo = [k for k in keys if k not in res]
    print("todo", len(todo), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 2))) as ex:
        for n, (k, v) in enumerate(ex.map(job, todo)):
            if v is not None:
                res[k] = v
            if n % 20 == 19:
                np.savez(CACHE, features=np.array(res, dtype=object)); print(n + 1, round(time.time() - t0), "s", flush=True)
    np.savez(CACHE, features=np.array(res, dtype=object)); print("finished", len(res), round(time.time() - t0), "s")
