import sys, time
from concurrent.futures import ThreadPoolExecutor as ProcessPoolExecutor
import numpy as np
import cv2; cv2.setNumThreads(1)
from custody.features_v2 import extract_features_v2
from custody.vision_loader import discover_triplets

def work(p):
    try:
        return p, extract_features_v2(p)
    except Exception as e:
        return p, repr(e)

if __name__ == "__main__":
    root = "E:/DF paper/data/vision"
    paths = []
    for t in discover_triplets(root):
        paths += [t["native"], t["whatsapp"], t["youtube"]]
    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(8) as ex:
        for p, f in ex.map(work, paths, chunksize=4):
            out[p] = f
    bad = {k: v for k, v in out.items() if isinstance(v, str)}
    print("done", len(out), "bad", len(bad), round(time.time() - t0), "s")
    for k, v in list(bad.items())[:5]: print(k, v)
    np.savez("data/vision/.feature_cache_v2.npz", features=np.array(out, dtype=object))
