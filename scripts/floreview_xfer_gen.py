"""Second-dataset transfer: FloreView natives (43 devices, never seen in training) pushed through the calibrated, GOP-matched
platform pipeline. Per clip: untampered control + the five edits of custody.forge.OPS. Features are the same as for VISION.
Resumable; writes work/floreview_feats.pkl.   GOP_VARIANT=minkey python scripts/floreview_xfer_gen.py"""
import os, sys, glob, time, pickle
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import av, numpy as np
import cv2; cv2.setNumThreads(1)
import custody.forge as F
from custody.features_v2 import extract_features_ext, pair_features_multi
from custody.local_features import decode_small, local_pair_features

sys.path.insert(0, "scripts")
os.environ.setdefault("GOP_VARIANT", "minkey")
from gen_gop_matched import make_args, VARIANT
from gen_forgeries import WA_SPEC

SRC = "E:/DF paper/data/floreview"
OUT = "E:/DF paper/work/forged_fv"
CACHE = "work/floreview_feats.pkl"


def clips():
    res = []
    for d in sorted(os.listdir(SRC)):
        fs = [f for f in glob.glob(f"{SRC}/{d}/*") if f.lower().endswith((".mp4", ".mov", ".3gp"))]
        if fs: res.append((d, fs[0].replace(chr(92), "/")))
    return res


def job(i, dev, nat, cl):
    F.x264_args = make_args(VARIANT)
    c = av.open(nat); cc = c.streams.video[0].codec_context; w, h = cc.width, cc.height; c.close()
    other = cl[(i + 1) % len(cl)][1]
    ps = {}
    for op in (None,) + tuple(F.OPS):
        p = f"{OUT}/{i:03d}_{op or 'none'}.mp4"
        if not os.path.exists(p):
            F.render_intermediate(nat, p, op, seed=5000 + i, other=other, target=F.wa_target_dims(w, h), direct=WA_SPEC)
        ps[op] = p
    plist = list(ps.values())
    ext = {p: extract_features_ext(p) for p in plist}
    pair = dict(zip(plist, pair_features_multi(nat, plist)))
    src = decode_small(nat)
    res = {"native_ext": extract_features_ext(nat)}
    for op, p in ps.items():
        res[op or "ctrl"] = dict(path=p, ext=ext[p], pair=pair[p], loc=local_pair_features(src, decode_small(p)))
    return dict(device=dev, native=nat, **res)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cl = clips()
    done = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) and os.path.getsize(CACHE) else {}
    t0 = time.time(); print(len(cl), "clips,", len(done), "done, variant", VARIANT, flush=True)
    for i, (dev, nat) in enumerate(cl):
        if i in done: continue
        try:
            done[i] = job(i, dev, nat, cl)
        except Exception as e:
            print("FAIL", i, dev, repr(e)[:120], flush=True); continue
        pickle.dump(done, open(CACHE + ".tmp", "wb")); os.replace(CACHE + ".tmp", CACHE)
        print(i + 1, "/", len(cl), "%.0fs" % (time.time() - t0), flush=True)
    print("finished", len(done))
