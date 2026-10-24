"""Established-method baselines for the FSI revision (new file; touches no existing output).

Four single-purpose detectors from the video-forensics literature, each reduced to a small feature vector per file and scored
by a shrinkage Mahalanobis distance to genuine WhatsApp files of other devices (leave-one-device-out, see eval_baselines.py):

  gop   - coding-structure check: GOP length, frame-type mix, packet-size pattern (re-encoding / platform-structure detection)
  dq    - double-quantization cue on decoded I-frames: periodicity of block-DCT coefficient histograms, first-digit (Benford) divergence
  prnu  - sensor-noise residual consistency between source and copy (PRNU-style normalized cross-correlation, global and per cell)
  temp  - temporal consistency: near-duplicate frames and frame-difference discontinuities (frame freeze / deletion cues)

Usage: python scripts/baselines_established.py [clip_index ...]   (no args = all clips, multiprocessing)
"""
import os, sys, pickle, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import av, cv2
from scipy.fft import dctn
from multiprocessing import Pool

os.environ.setdefault("FEATSET", "v2"); os.environ.setdefault("PAIR", "1"); os.environ.setdefault("LOCAL", "0")
sys.path.insert(0, "scripts")
import eval_v2 as E

NF = 180
cv2.setNumThreads(1)
TIERS = tuple(os.environ["BASE_TIERS"].split(",")) if os.environ.get("BASE_TIERS") else ("naive", "platform", "adaptive", "adaptive_drift_only", "adaptive_gbm", "simctl")
OUTDIR = os.environ.get("BASE_OUT", "work/baselines_v2")
OUT = "work/baselines_feats.pkl"


def decode(path, n=NF, size=None):
    """Streaming decode (memory-lean). Returns packets, picture types, key flags, <=4 full key frames,
    every 6th frame's centre crop, and the 1/4-scale frame-difference series. If size=(w,h) frames are resized first (native side)."""
    pk = []
    c = av.open(str(path)); s = c.streams.video[0]
    for p in c.demux(s):
        if p.size:
            pk.append((p.size, bool(p.is_keyframe)))
        if len(pk) >= n:
            break
    c.close()
    c = av.open(str(path)); s = c.streams.video[0]; s.thread_type = "NONE"
    pt, kf, keyfr, crops, d, prev, hw = [], [], [], [], [], None, None
    for t, fr in enumerate(c.decode(s)):
        f = fr.to_ndarray(format="gray")
        if size is not None and (f.shape[1], f.shape[0]) != size:
            f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        hw = f.shape
        pt.append({1: "I", 2: "P", 3: "B"}.get(int(fr.pict_type), "?")); kf.append(bool(fr.key_frame))
        if fr.key_frame and len(keyfr) < 4:
            keyfr.append(f.copy())
        if t % 6 == 0:
            h, w = f.shape; ch, cw = min(h, 384), min(w, 512); y0, x0 = (h - ch) // 2, (w - cw) // 2
            crops.append(f[y0:y0 + ch, x0:x0 + cw].copy())
        sm = cv2.resize(f, (f.shape[1] // 4, f.shape[0] // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            d.append(float(np.abs(sm - prev).mean()))
        prev = sm
        if t + 1 >= n:
            break
    c.close()
    if not keyfr:
        keyfr = [f]
    return dict(pt=pt, kf=kf, pk=pk, keyfr=keyfr, crops=crops, d=np.array(d), hw=hw)


def f_gop(pt, kf, pk):
    sz = np.array([p[0] for p in pk], float); key = np.array([p[1] for p in pk])
    ki = np.flatnonzero(key); gop = np.diff(ki) if len(ki) > 1 else np.array([float(len(sz))])
    fb = np.mean([t == "B" for t in pt]) if pt else 0.0; fi = np.mean([t == "I" for t in pt]) if pt else 0.0
    si = sz[key].mean() if key.any() else sz.mean(); sp = sz[~key] if (~key).any() else sz
    lag1 = np.corrcoef(sz[:-1], sz[1:])[0, 1] if len(sz) > 3 and sz.std() > 0 else 0.0
    return np.array([np.log1p(gop.mean()), gop.std(), fb, fi, np.log(si), np.log(sp.mean() + 1), np.log(si) - np.log(sp.mean() + 1),
                     sp.std() / (sp.mean() + 1e-6), 0.0 if np.isnan(lag1) else lag1, np.log(sz.mean())])


def _benford_kl(v):
    v = np.abs(v[np.abs(v) >= 1]).astype(float)
    if len(v) < 50:
        return 0.0
    d = (v / 10 ** np.floor(np.log10(v))).astype(int).clip(1, 9)
    p = np.bincount(d, minlength=10)[1:10] / len(d); q = np.log10(1 + 1 / np.arange(1, 10))
    return float(np.sum(p * np.log((p + 1e-9) / q)))


def f_dq(keyfr):
    per, ben, zer, mag = [], [], [], []
    pos = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (2, 1)]
    for kfm in keyfr:
        f = kfm.astype(np.float32); h, w = f.shape; ch, cw = min(h, 256) // 8 * 8, min(w, 256) // 8 * 8
        y0, x0 = (h - ch) // 2 // 8 * 8, (w - cw) // 2 // 8 * 8
        b = f[y0:y0 + ch, x0:x0 + cw].reshape(ch // 8, 8, cw // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
        D = dctn(b - 128.0, axes=(1, 2), norm="ortho")
        for (u, v) in pos:
            c = np.rint(D[:, u, v]).astype(int); hist = np.bincount(np.clip(c, -30, 30) + 30, minlength=61).astype(float)
            hist /= hist.sum() + 1e-9; F = np.abs(np.fft.rfft(hist))[1:]
            per.append(F.max() / (F.mean() + 1e-9))
        ac = np.rint(D.reshape(len(D), 64)[:, 1:]).ravel()
        ben.append(_benford_kl(ac)); zer.append(np.mean(ac == 0)); mag.append(np.log1p(np.abs(ac).mean()))
    return np.array([np.mean(per), np.std(per), np.mean(ben), np.mean(zer), np.mean(mag)])


def _resid(f):
    f = f.astype(np.float32)
    return f - cv2.GaussianBlur(f, (5, 5), 1.2)


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def f_prnu(src_crops, tgt_crops):
    glob, cellmin = [], []
    for sa, ta in zip(src_crops, tgt_crops):
        ch, cw = ta.shape
        if sa.shape != ta.shape:
            continue
        a = _resid(sa); b = _resid(ta)
        glob.append(_ncc(a, b))
        cs = []
        for yy in range(0, ch - ch // 3 + 1, ch // 3):
            for xx in range(0, cw - cw // 4 + 1, cw // 4):
                cs.append(_ncc(a[yy:yy + ch // 3, xx:xx + cw // 4], b[yy:yy + ch // 3, xx:xx + cw // 4]))
        cellmin.append(min(cs))
    glob = np.array(glob); w5 = np.convolve(glob, np.ones(5) / 5, mode="valid") if len(glob) >= 5 else glob
    return np.array([glob.mean(), np.percentile(glob, 10), np.mean(cellmin), np.median(w5) - w5.min()])


def f_temp(d):
    med = np.median(d) + 1e-6; dup = d < 0.05 * med
    run = best = 0
    for x in dup:
        run = run + 1 if x else 0; best = max(best, run)
    mad = np.median(np.abs(d - np.median(d))) * 1.4826 + 1e-3
    z = (d - np.median(d)) / mad
    return np.array([dup.mean(), np.log1p(best), np.log1p(max(z.max(), 0)), np.log1p(max(-z.min(), 0))])


def feats(src_cache, native, path):
    r = decode(path); h, w = r["hw"]
    key = (native, w, h)
    if key not in src_cache:
        src_cache.clear(); src_cache[key] = decode(native, size=(w, h))["crops"]
    return dict(gop=f_gop(r["pt"], r["kf"], r["pk"]), dq=f_dq(r["keyfr"]), prnu=f_prnu(src_cache[key], r["crops"]), temp=f_temp(r["d"]))


def job(i):
    t = E.T[i]; cache = {}; out = {}
    try:
        if os.environ.get("BASE_TIERS") is None:
            out[("genuine", i, None)] = feats(cache, t["native"], t["whatsapp"])
        for tier in TIERS:
            for op in ([None] if tier.startswith("simctl") else E.OPS):
                p = E.forged_path(tier, i, op)
                if os.path.exists(p):
                    out[(tier, i, op)] = feats(cache, t["native"], p)
    except Exception as e:
        print("clip", i, "ERR", repr(e)[:120], flush=True)
    return out


def run_clip(i):
    p = f"{OUTDIR}/clip_{i:03d}.pkl"
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return i, 0
    r = job(i)
    tmp = p + ".tmp"
    pickle.dump(r, open(tmp, "wb")); os.replace(tmp, p)
    return i, len(r)


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    args = [int(a) for a in sys.argv[1:]]
    if args:
        t0 = time.time(); r = job(args[0]); print(len(r), "files in %.1fs" % (time.time() - t0))
    else:
        t0 = time.time(); todo = list(range(len(E.T)))
        with Pool(int(os.environ.get("WORKERS", 2)), maxtasksperchild=1) as pool:
            for n, (i, k) in enumerate(pool.imap_unordered(run_clip, todo)):
                print(n + 1, "/", len(todo), "clip", i, k, "files %.0fs" % (time.time() - t0), flush=True)
        print("done")
