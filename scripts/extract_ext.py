"""Extended features for every file on disk (genuine VISION + all forged/chain files); resumable."""
import glob, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import cv2; cv2.setNumThreads(1)
import numpy as np
from custody.features_v2 import extract_features_ext
from custody.vision_loader import discover_triplets

CACHE = "work/features_ext.npz"

def work(p):
    for a in range(3):
        try:
            return p, extract_features_ext(p)
        except Exception as e:
            time.sleep(5)
    return p, None

if __name__ == "__main__":
    paths = []
    for t in discover_triplets("E:/DF paper/data/vision"):
        paths += [t["native"], t["whatsapp"], t["youtube"]]
    for pat in sys.argv[1:] or ["work/forged/*/*.mp4"]:
        paths += sorted(glob.glob(pat))
    done = np.load(CACHE, allow_pickle=True)["features"].item() if os.path.exists(CACHE) else {}
    todo = [p for p in paths if p not in done]
    print("todo", len(todo), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 4))) as ex:
        for n, (p, f) in enumerate(ex.map(work, todo)):
            if f is not None:
                done[p] = f
            if n % 100 == 99:
                np.savez(CACHE, features=np.array(done, dtype=object)); print(n + 1, round(time.time() - t0), "s", flush=True)
    np.savez(CACHE, features=np.array(done, dtype=object))
    print("finished", len(done), "missing", len([p for p in todo if p not in done]), round(time.time() - t0), "s")
