"""Features for the GOP-matched tiers (platform_gop, simctl_gop): file features, source-vs-copy pair features, local features.
Writes work/gop_matched_feats.pkl without touching the existing feature caches."""
import os, sys, time, pickle
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import cv2; cv2.setNumThreads(1)
from concurrent.futures import ThreadPoolExecutor
from custody.features_v2 import extract_features_ext, pair_features_multi
from custody.local_features import decode_small, local_pair_features
from custody.vision_loader import discover_triplets
from custody.forge import OPS

T = discover_triplets("E:/DF paper/data/vision")
OUT = "work/gop_matched_feats.pkl"


def paths(i):
    return [f"work/forged/simctl_gop/{i:03d}_none.mp4"] + [f"work/forged/platform_gop/{i:03d}_{op}.mp4" for op in OPS]


def job(i):
    ps = [p for p in paths(i) if os.path.exists(p)]
    if not ps:
        return {}
    nat = T[i]["native"]; res = {}
    ext = {p: extract_features_ext(p) for p in ps}
    pair = dict(zip(ps, pair_features_multi(nat, ps)))
    src = decode_small(nat)
    for p in ps:
        res[p] = dict(ext=ext[p], pair=pair[p], loc=local_pair_features(src, decode_small(p)), native=nat)
    return res


if __name__ == "__main__":
    done = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) and os.path.getsize(OUT) > 0 else {}
    todo = [i for i in range(len(T)) if not all(p in done for p in paths(i) if os.path.exists(p))]
    t0 = time.time(); print("clips todo", len(todo), flush=True)
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 2))) as ex:
        for n, r in enumerate(ex.map(job, todo)):
            done.update(r)
            if n % 10 == 9:
                pickle.dump(done, open(OUT + ".tmp", "wb")); os.replace(OUT + ".tmp", OUT); print(n + 1, "/", len(todo), "%.0fs" % (time.time() - t0), flush=True)
    pickle.dump(done, open(OUT + ".tmp", "wb")); os.replace(OUT + ".tmp", OUT); print("done", len(done))
