"""Strongest platform-aware tier: the forger also matches the platform's fixed 30-frame GOP (no extra intra frames).

The baseline study (scripts/baselines_established.py) showed that our standard platform-aware encodes contain extra I-type frames
that real WhatsApp files do not (min-keyint 25 and lookahead frame-type decisions), so a coding-structure check separates them from real
files. This tier removes that difference with documented x264 options and re-runs the same eight forgeries.

  python scripts/gen_gop_matched.py test <clip> <variant>    -> encode one control file and report its frame types
  python scripts/gen_gop_matched.py all                      -> tier 'platform_gop' (5 edits x 86 clips) + control 'simctl_gop'
"""
import os, sys, time, collections
import numpy as np, av
from concurrent.futures import ThreadPoolExecutor
import custody.forge as F

sys.path.insert(0, "scripts")
from gen_forgeries import WA_SPEC, T, other_clip, OUT

VARIANTS = {
    "minkey": "keyint={k}:min-keyint={k}:bframes=0:deblock=0:0:scenecut=0",
    "minkey_nola": "keyint={k}:min-keyint={k}:bframes=0:deblock=0:0:scenecut=0:rc-lookahead=0:no-mbtree=1",
    "minkey_nola_cabac": "keyint={k}:min-keyint={k}:bframes=0:deblock=0:0:scenecut=0:rc-lookahead=0:no-mbtree=1:no-cabac=1",
}
VARIANT = os.environ.get("GOP_VARIANT", "minkey_nola")


def make_args(variant):
    def x264_args(crf=28.0, preset="medium", profile="baseline", keyint=30, bframes=0, deblock=(0, 0), **kw):
        xp = VARIANTS[variant].format(k=int(keyint))
        return ["-threads", "2", "-c:v", "libx264", "-preset", preset, "-profile:v", profile, "-crf", f"{crf:.2f}",
                "-sc_threshold", "0", "-x264-params", xp, "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart"]
    return x264_args


def types(path):
    c = av.open(path); s = c.streams.video[0]
    pt = [int(fr.pict_type) for fr in c.decode(s)]; c.close()
    kf = [i for i, t in enumerate(pt) if t == 1]
    return len(pt), collections.Counter(pt), np.diff(kf).tolist()[:8]


def encode_one(tier, i, op, variant=VARIANT):
    F.x264_args = make_args(variant)
    t = T[i]
    c = av.open(t["native"]); cc = c.streams.video[0].codec_context; w, h = cc.width, cc.height; c.close()
    os.makedirs(f"{OUT}/{tier}", exist_ok=True)
    out = f"{OUT}/{tier}/{i:03d}_{op or 'none'}.mp4"
    F.render_intermediate(t["native"], out, op, seed=1000 + i, other=other_clip(i), target=F.wa_target_dims(w, h), direct=WA_SPEC)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "test":
        i = int(sys.argv[2]); var = sys.argv[3]
        out = encode_one("gop_test_" + var, i, None, var)
        print(var, i, types(out))
        real = types(T[i]["whatsapp"]); print("real WhatsApp:", real)
    else:
        jobs = [("simctl_gop", i, None) for i in range(len(T))] + [("platform_gop", i, op) for i in range(len(T)) for op in F.OPS]
        t0 = time.time()
        def run(j):
            if os.path.exists(f"{OUT}/{j[0]}/{j[1]:03d}_{j[2] or 'none'}.mp4"):
                return 0
            encode_one(*j); return 1
        with ThreadPoolExecutor(int(os.environ.get("WORKERS", 2))) as ex:
            for n, r in enumerate(ex.map(run, jobs)):
                if n % 40 == 39:
                    print(n + 1, "/", len(jobs), "%.0fs" % (time.time() - t0), flush=True)
        print("done")
