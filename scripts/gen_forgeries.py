"""Generate real forged files (naive / platform-aware) and the untampered pipeline control."""
import os, sys, time, json
from concurrent.futures import ThreadPoolExecutor
import av, numpy as np
from custody.forge import render_intermediate, wa_target_dims, OPS, EXTRA_OPS
from custody.features_v2 import extract_features_v2
from custody.vision_loader import discover_triplets

ROOT = "E:/DF paper/data/vision"
OUT = "E:/DF paper/work/forged"
WA_SPEC = dict(crf=20.5, preset="veryfast", profile="baseline", keyint=30, bframes=0)
NAIVE_SPEC = dict(crf=23.0, preset="medium", profile="high", keyint=250, bframes=3, scenecut=40)
T = discover_triplets(ROOT)

def other_clip(i):
    t = T[i]
    cands = [j for j, u in enumerate(T) if u["device"] == t["device"] and j != i]
    return T[(cands[i % len(cands)] if cands else (i + 1) % len(T))]["native"]

def job(args):
    for attempt in range(4):
        try:
            return _job(args)
        except Exception as e:
            print('retry', args, attempt, repr(e)[:80], flush=True)
            time.sleep(10)
    raise RuntimeError(f'failed {args}')


def _job(args):
    tier, i, op = args
    t = T[i]
    c = av.open(t["native"]); cc = c.streams.video[0].codec_context; w, h = cc.width, cc.height; c.close()
    os.makedirs(f"{OUT}/{tier}", exist_ok=True)
    out = f"{OUT}/{tier}/{i:03d}_{op or 'none'}.mp4"
    if tier == "naive":
        render_intermediate(t["native"], out, op, seed=1000 + i, other=other_clip(i), target=(w, h), direct=NAIVE_SPEC)
    else:
        render_intermediate(t["native"], out, op, seed=1000 + i, other=other_clip(i), target=wa_target_dims(w, h), direct=WA_SPEC)
    return (tier, i, op), extract_features_v2(out)

if __name__ == "__main__":
    if os.environ.get("EXTRA"):
        jobs = [("platform_x", i, op) for i in range(len(T)) for op in EXTRA_OPS]
        cache0 = "work/forged_features_x.npz"
    else:
        jobs = [("simctl", i, None) for i in range(len(T))]
        jobs += [(tier, i, op) for tier in ("platform", "naive") for i in range(len(T)) for op in OPS]
        cache0 = "work/forged_features.npz"
    cache = cache0
    done = {}
    if os.path.exists(cache):
        done = np.load(cache, allow_pickle=True)["features"].item()
    todo = [j for j in jobs if j not in done]
    print("todo", len(todo), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(int(os.environ.get('WORKERS', 2))) as ex:
        for n, (k, f) in enumerate(ex.map(job, todo)):
            done[k] = f
            if n % 50 == 49:
                np.savez(cache, features=np.array(done, dtype=object)); print(n + 1, round(time.time() - t0), "s", flush=True)
    np.savez(cache, features=np.array(done, dtype=object))
    print("finished", len(done), round(time.time() - t0), "s")
