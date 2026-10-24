"""Extend the adaptive search: continue hill-climbing 20 more trials from the stored best setting (12 clips x 5 edits)."""
import os, sys, json, time, shutil, threading
from concurrent.futures import ThreadPoolExecutor
import av, numpy as np
from custody.forge import render_intermediate, encode, wa_target_dims, OPS
from custody.features_v2 import extract_features_ext, _gray_frames, _pair_from_frames
import cv2
sys.path.insert(0, "scripts")
import eval_v2 as E
import gen_adaptive as G
from gen_forgeries import other_clip, T

EXTRA_TRIALS = int(os.environ.get("EXTRA_TRIALS", 20))
meta = json.load(open("work/adaptive_meta.json"))
clips = sorted({int(k.split("_")[0]) for k in meta})[::3][:12]
SCR = G.SCR


def job(a):
    i, op = a
    for att in range(4):
        try:
            return _job(a)
        except Exception as e:
            print("retry", a, repr(e)[:70], flush=True); time.sleep(15)
    return a, None


def _job(a):
    i, op = a
    m = meta[f"{i}_{op}"]; det = G.DETS[E.DEV[i]]; t = T[i]
    c = av.open(t["native"]); cc = c.streams.video[0].codec_context; w, h = cc.width, cc.height; c.close()
    tag = threading.get_ident(); inter = f"{SCR}/b{tag}.mkv"
    render_intermediate(t["native"], inter, op, seed=1000 + i, other=other_clip(i), target=wa_target_dims(w, h))
    src_g = _gray_frames(t["native"], 180); cache = {}
    def ev(p):
        out = f"{SCR}/bo{tag}.mp4"; encode(inter, out, **p)
        f = extract_features_ext(out, local=False)[:E.NCOL]; tg = _gray_frames(out, 180); sz = (tg[0].shape[1], tg[0].shape[0])
        if sz not in cache: cache[sz] = [x if (x.shape[1], x.shape[0]) == sz else cv2.resize(x, sz, interpolation=cv2.INTER_AREA) for x in src_g]
        v = np.r_[f, _pair_from_frames(cache[sz], tg)]
        return float(det.scores(E.N[[i]], v[None, :])["full_system"][0])
    p = dict(m["params"]); p["deblock"] = tuple(p.get("deblock", (0, 0))); p["maxrate_kbps"] = p.get("maxrate_kbps")
    best = m["best"]; rng = np.random.default_rng(9000 + i); trace = [best]
    for _ in range(EXTRA_TRIALS):
        q = G.perturb(p, rng)
        try: s = ev(q)
        except Exception: s = np.inf
        if s < best: best, p = s, q
        trace.append(best)
    return a, trace


if __name__ == "__main__":
    G.DETS = E.fold_detectors()
    res = json.load(open("work/budget_trace.json")) if os.path.exists("work/budget_trace.json") else {}
    jobs = [(i, op) for i in clips for op in OPS if f"{i}_{op}" in meta and f"{i}_{op}" not in res]
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 2))) as ex:
        for n, (a, tr) in enumerate(ex.map(job, jobs)):
            if tr: res[f"{a[0]}_{a[1]}"] = tr
            if n % 5 == 4: json.dump(res, open("work/budget_trace.json", "w")); print(n + 1, flush=True)
    json.dump(res, open("work/budget_trace.json", "w")); print("finished", len(res))
