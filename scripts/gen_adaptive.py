"""White-box adaptive attacker: searches real x264 encoder settings to minimise the deployed
full-system score of its own forged file (against the detector fitted on all other devices)."""
import os, sys, time, json, threading
from concurrent.futures import ThreadPoolExecutor
import av, numpy as np
from custody.forge import render_intermediate, encode, wa_target_dims, OPS
from custody.features_v2 import extract_features_ext, _gray_frames, _pair_from_frames
from custody.local_features import decode_small, local_pair_features
import joblib
import cv2
sys.path.insert(0, "scripts")
import eval_v2 as E
from gen_forgeries import other_clip, WA_SPEC, T

SCR = "C:/Users/Ridan/AppData/Local/Temp/claude/E--DF-paper/4ac6b769-ad6d-461e-b926-3277c9e8c3af/scratchpad/adapt"
OBJ = os.environ.get("OBJ", "full_system")
TIER = "adaptive" if OBJ == "full_system" else f"adaptive_{OBJ}"
OUT = f"E:/DF paper/work/forged/{TIER}"
os.makedirs(SCR, exist_ok=True); os.makedirs(OUT, exist_ok=True)
PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast"]
K_RANDOM, K_REFINE = int(os.environ.get("K_RANDOM", 6)), int(os.environ.get("K_REFINE", 4))
STRIDE = int(os.environ.get("STRIDE", 2))
DETS = None
LRC = json.load(open("work/fusion_coefs.json")) if OBJ == "fusion" else None
GBM = {}


def sample(rng):
    return dict(crf=float(rng.uniform(16, 26)), preset=str(rng.choice(PRESETS)), profile="baseline",
                keyint=int(rng.choice([24, 30, 48, 60])), bframes=0,
                deblock=(int(rng.integers(-1, 2)), int(rng.integers(-1, 2))),
                pre_noise=float(rng.choice([0, 0, 2, 4, 6])), denoise=float(rng.choice([0, 0, 1, 2])),
                maxrate_kbps=rng.choice([None, 1200, 1800, 2500]), refs=int(rng.integers(1, 4)))


def perturb(p, rng):
    q = dict(p)
    k = rng.choice(["crf", "preset", "keyint", "deblock", "pre_noise", "denoise", "maxrate_kbps", "refs"])
    r = sample(rng)
    q[k] = r[k]
    if k != "crf" and rng.random() < 0.5:
        q["crf"] = float(np.clip(p["crf"] + rng.normal(0, 1.2), 14, 28))
    return q


def job(args):
    for att in range(5):
        try:
            return _job(args)
        except Exception as e:
            print('retry', args, att, repr(e)[:80], flush=True)
            time.sleep(20)
    raise RuntimeError(f'failed {args}')


def _job(args):
    i, op = args
    d = E.DEV[i]
    det = DETS[d]
    t = T[i]
    c = av.open(t["native"]); cc = c.streams.video[0].codec_context; w, h = cc.width, cc.height; c.close()
    tag = threading.get_ident()
    inter = f"{SCR}/i{tag}.mkv"
    render_intermediate(t["native"], inter, op, seed=1000 + i, other=other_clip(i), target=wa_target_dims(w, h))
    rng = np.random.default_rng(5000 + i * 7 + OPS.index(op))
    trace = []

    src_g = _gray_frames(t["native"], 180)
    src_small = decode_small(t["native"]) if OBJ == "gbm" else None
    clf = joblib.load(f"work/gbm_folds/{d}.joblib") if OBJ == "gbm" else None
    size_cache = {}

    def evaluate(p):
        out = f"{SCR}/o{tag}.mp4"
        encode(inter, out, **p)
        f = extract_features_ext(out, local=False)[:E.NCOL]
        tg = _gray_frames(out, 180)
        sz = (tg[0].shape[1], tg[0].shape[0])
        if sz not in size_cache:
            size_cache[sz] = [x if (x.shape[1], x.shape[0]) == sz else cv2.resize(x, sz, interpolation=cv2.INTER_AREA) for x in src_g]
        v = np.r_[f, _pair_from_frames(size_cache[sz], tg)] if E.PAIR else f
        if OBJ == "gbm":
            v = np.r_[v, local_pair_features(src_small, decode_small(out))]
            return float(clf.predict_proba(v[None, :])[0, 1]), v
        sa = det.scores(E.N[[i]], v[None, :])
        if OBJ == "fusion":
            s = LRC[E.DEV[i]]["intercept"] + sum(c * sa[k][0] for c, k in zip(LRC[E.DEV[i]]["coef"], LRC[E.DEV[i]]["cols"]))
        else:
            s = sa[OBJ][0]
        return float(s), v

    best_p = dict(WA_SPEC); best_s, best_f = evaluate(best_p)
    start = best_s
    trace.append(start)
    import shutil
    best_file = f"{SCR}/best{tag}.mp4"
    shutil.copyfile(f"{SCR}/o{tag}.mp4", best_file)
    cands = [sample(rng) for _ in range(K_RANDOM)]
    for step in range(K_RANDOM + K_REFINE):
        p = cands[step] if step < K_RANDOM else perturb(best_p, rng)
        try:
            s, f = evaluate(p)
        except Exception:
            continue
        trace.append(s)
        if s < best_s:
            best_s, best_f, best_p = s, f, p
            shutil.copyfile(f"{SCR}/o{tag}.mp4", best_file)
    shutil.copyfile(best_file, f"{OUT}/{i:03d}_{op}.mp4")
    return (TIER, i, op), best_f, dict(start=start, best=best_s, params={k: (v if not isinstance(v, np.generic) else v.item()) for k, v in best_p.items()}, trace=trace)


if __name__ == "__main__":
    DETS = E.fold_detectors()
    ops = os.environ.get("OPS", ",".join(OPS)).split(",")
    cache = f"work/{TIER}_features.npz"; meta_f = f"work/{TIER}_meta.json"
    done = np.load(cache, allow_pickle=True)["features"].item() if os.path.exists(cache) else {}
    meta = json.load(open(meta_f)) if os.path.exists(meta_f) else {}
    todo = [(i, op) for i in range(0, len(T), STRIDE) for op in ops if (TIER, i, op) not in done]
    print("todo", len(todo), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 3))) as ex:
        for n, (k, f, m) in enumerate(ex.map(job, todo)):
            done[k] = f; meta[f"{k[1]}_{k[2]}"] = m
            if n % 10 == 9:
                np.savez(cache, features=np.array(done, dtype=object)); json.dump(meta, open(meta_f, "w"))
                print(n + 1, round(time.time() - t0), "s", flush=True)
    np.savez(cache, features=np.array(done, dtype=object)); json.dump(meta, open(meta_f, "w"))
    print("finished", len(done), round(time.time() - t0), "s")
