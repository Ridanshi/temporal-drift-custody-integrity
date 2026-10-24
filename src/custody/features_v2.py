"""Richer custody features.

Two channels per video:
  declared  (read from container / stream headers, no pixel decode):
      log10 container bit rate, log10 pixel count, frame rate, profile ordinal
  measured  (from decoded pixels and packet sizes):
      log noise residual std, log residual kurtosis, spatial noise inconsistency,
      log 8x8 blockiness, DCT high-frequency ratio, log temporal noise,
      log10 measured bits/pixel, log10 I/non-I packet-size ratio

All pixel statistics are computed on a central crop of at most 512x384 aligned to the
16-pixel macroblock grid, over the first `max_frames` frames, so that files of different
length are comparable.
"""
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.fft import dctn
from scipy.stats import kurtosis

DECLARED_NAMES = ["log_bitrate", "log_pixels", "fps", "profile"]
MEASURED_NAMES = [
    "noise_std", "noise_kurt", "noise_incons", "blockiness", "hf_ratio",
    "log_temporal_noise", "log_bpp", "log_ifr",
]
FEATURE_NAMES = DECLARED_NAMES + MEASURED_NAMES
N_DECLARED = len(DECLARED_NAMES)

_PROFILE_ORD = {"baseline": 0, "constrained baseline": 0, "main": 1, "extended": 1, "high": 2}


def _crop_box(h, w, ch=384, cw=512):
    ch, cw = min(ch, h), min(cw, w)
    y0 = ((h - ch) // 2) // 16 * 16
    x0 = ((w - cw) // 2) // 16 * 16
    ch, cw = ch // 16 * 16, cw // 16 * 16
    return y0, x0, ch, cw


def _spatial_stats(g):
    """g: uint8 gray crop with sides multiple of 16."""
    med = cv2.medianBlur(g, 3)
    res = g.astype(np.float32) - med.astype(np.float32)
    h, w = g.shape
    gy, gx = 3, 4
    cells = res[: h // gy * gy, : w // gx * gx].reshape(gy, h // gy, gx, w // gx).std(axis=(1, 3)).ravel()
    noise_std = float(np.median(cells))
    incons = float(cells.std() / (cells.mean() + 1e-6))
    kurt = float(np.nan_to_num(kurtosis(res.ravel(), fisher=True), nan=0.0)) if res.std() > 1e-6 else 0.0
    f = g.astype(np.float32)
    dx = np.abs(np.diff(f, axis=1))
    on = dx[:, 7::8].mean()
    off = np.delete(dx, np.arange(7, dx.shape[1], 8), axis=1).mean()
    blockiness = float(on / (off + 1e-6))
    blk = f.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    c = dctn(blk, axes=(2, 3), norm="ortho")
    e = c ** 2
    uv = np.add.outer(np.arange(8), np.arange(8))
    tot = e.sum(axis=(0, 1)) - 0
    tot[0, 0] = 0.0
    hf = float(tot[uv >= 8].sum() / (tot.sum() + 1e-9))
    return noise_std, kurt, incons, blockiness, hf


def extract_features_v2(path, max_frames=180, spatial_every=6):
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(path)
    c = av.open(path)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    cc = s.codec_context
    w, h = cc.width, cc.height
    bit_rate = c.bit_rate or s.bit_rate or 0
    if not bit_rate and c.duration:
        bit_rate = Path(path).stat().st_size * 8 / (c.duration / 1e6)
    fps = float(s.average_rate) if s.average_rate else 30.0
    prof = _PROFILE_ORD.get(str(cc.profile).lower(), 1)
    declared = [np.log10(max(bit_rate, 1)), np.log10(w * h), fps, float(prof)]

    y0, x0, ch, cw = _crop_box(h, w)
    sizes, is_key, frames_seen = [], [], 0
    prev, tdiff, spat = None, [], []
    for packet in c.demux(s):
        if packet.size == 0:
            continue
        for frame in packet.decode():
            if frames_seen >= max_frames:
                break
            g = frame.to_ndarray(format="gray")[y0 : y0 + ch, x0 : x0 + cw]
            if prev is not None:
                tdiff.append(float(np.abs(g.astype(np.float32) - prev).std()))
            prev = g.astype(np.float32)
            if frames_seen % spatial_every == 0:
                spat.append(_spatial_stats(np.ascontiguousarray(g)))
            sizes.append(packet.size)
            is_key.append(bool(packet.is_keyframe))
            frames_seen += 1
        if frames_seen >= max_frames:
            break
    c.close()
    if frames_seen < 30:
        raise ValueError(f"too few frames in {path}: {frames_seen}")
    spat = np.array(spat)
    sizes = np.array(sizes, dtype=float)
    is_key = np.array(is_key)
    bpp = sizes.sum() * 8 / (frames_seen * w * h)
    i_sz = sizes[is_key].mean() if is_key.any() else sizes.mean()
    n_sz = sizes[~is_key].mean() if (~is_key).any() else i_sz
    sm = spat.mean(axis=0)
    measured = [
        float(np.log10(sm[0] + 1e-6)),
        float(np.log10(1.0 + max(sm[1], 0.0))),
        float(sm[2]),
        float(np.log10(sm[3] + 1e-6)),
        float(sm[4]),
        float(np.log10(np.mean(tdiff) + 0.01)),
        float(np.log10(bpp + 1e-9)),
        float(np.log10(i_sz / (n_sz + 1e-9) + 1e-9)),
    ]
    return np.array(declared + measured, dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# Extended feature set: v2 (12 global features) + 6 local-inconsistency features.
# ---------------------------------------------------------------------------------------------
from scipy.spatial import cKDTree  # noqa: E402

LOCAL_NAMES = ["spatial_outlier", "noise_tjump", "log_tdiff_minfrac", "log_tdiff_burst", "cell_td_outlier", "dup_patch"]
FEATURE_NAMES_EXT = FEATURE_NAMES + LOCAL_NAMES
DUP_FRAMES = (30, 90, 150)


def _robust_z(x):
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med)) + 1e-2
    return (x - med) / mad


def _cell_logstd(g, gy=6, gx=8):
    res = g.astype(np.float32) - cv2.medianBlur(g, 3).astype(np.float32)
    h, w = g.shape
    c = res[: h // gy * gy, : w // gx * gx].reshape(gy, h // gy, gx, w // gx).std(axis=(1, 3))
    return np.log(c.ravel() + 1e-3)


def _dup_patch(g, blk=16, stride=8, min_sep=48):
    f = g.astype(np.float32)
    h, w = f.shape
    ys = np.arange(0, h - blk + 1, stride)
    xs = np.arange(0, w - blk + 1, stride)
    idx = [(y, x) for y in ys for x in xs]
    stack = np.stack([f[y:y + blk, x:x + blk] for y, x in idx])
    c = dctn(stack, axes=(1, 2), norm="ortho")
    zz = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2), (0, 3), (3, 0)]
    d = np.stack([c[:, u, v] for u, v in zz], axis=1)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-3)
    tree = cKDTree(d)
    dist, nn = tree.query(d, k=10)
    pos = np.array(idx, dtype=float)
    best = np.full(len(idx), np.inf)
    for j in range(1, 10):
        far = np.linalg.norm(pos - pos[nn[:, j]], axis=1) >= min_sep
        best = np.where(far & np.isinf(best), dist[:, j], best)
    best = np.where(np.isinf(best), np.nanmedian(dist[:, 1:]), best)
    return float(np.log10(np.percentile(best, 2) / (np.median(best) + 1e-6) + 1e-6))


def extract_features_ext(path, max_frames=180, spatial_every=6, cell_every=3, local=True):
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(path)
    c = av.open(path)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    s.codec_context.thread_count = 2
    cc = s.codec_context
    w, h = cc.width, cc.height
    bit_rate = c.bit_rate or s.bit_rate or 0
    if not bit_rate and c.duration:
        bit_rate = Path(path).stat().st_size * 8 / (c.duration / 1e6)
    fps = float(s.average_rate) if s.average_rate else 30.0
    prof = _PROFILE_ORD.get(str(cc.profile).lower(), 1)
    declared = [np.log10(max(bit_rate, 1)), np.log10(w * h), fps, float(prof)]
    y0, x0, ch, cw = _crop_box(h, w)
    sizes, is_key, n = [], [], 0
    prev, tdiff, spat, cells, cell_td, dups = None, [], [], [], [], []
    gy, gx = 6, 8
    for packet in c.demux(s):
        if packet.size == 0:
            continue
        for frame in packet.decode():
            if n >= max_frames:
                break
            g = np.ascontiguousarray(frame.to_ndarray(format="gray")[y0 : y0 + ch, x0 : x0 + cw])
            gf = g.astype(np.float32)
            if prev is not None:
                dd = np.abs(gf - prev)
                tdiff.append(float(dd.std()))
                if local:
                    cell_td.append(dd[: ch // gy * gy, : cw // gx * gx].reshape(gy, ch // gy, gx, cw // gx).mean(axis=(1, 3)).ravel())
            prev = gf
            if n % spatial_every == 0:
                spat.append(_spatial_stats(g))
            if local and n % cell_every == 0:
                cells.append(_cell_logstd(g))
            if local and n in DUP_FRAMES:
                dups.append(_dup_patch(g))
            sizes.append(packet.size)
            is_key.append(bool(packet.is_keyframe))
            n += 1
        if n >= max_frames:
            break
    c.close()
    if n < 30:
        raise ValueError(f"too few frames in {path}: {n}")
    spat = np.array(spat)
    sizes = np.array(sizes, dtype=float)
    is_key = np.array(is_key)
    bpp = sizes.sum() * 8 / (n * w * h)
    i_sz = sizes[is_key].mean() if is_key.any() else sizes.mean()
    n_sz = sizes[~is_key].mean() if (~is_key).any() else i_sz
    sm = spat.mean(axis=0)
    measured = [
        float(np.log10(sm[0] + 1e-6)), float(np.log10(1.0 + max(sm[1], 0.0))), float(sm[2]),
        float(np.log10(sm[3] + 1e-6)), float(sm[4]), float(np.log10(np.mean(tdiff) + 0.01)),
        float(np.log10(bpp + 1e-9)), float(np.log10(i_sz / (n_sz + 1e-9) + 1e-9)),
    ]
    if not local:
        return np.array(declared + measured, dtype=np.float64)
    cells = np.array(cells)
    sp_out = np.array([np.max(np.abs(_robust_z(r))) for r in cells])
    nlev = cells.mean(axis=1)
    dn = np.diff(nlev)
    jump = np.max(np.abs(dn)) / (1.4826 * np.median(np.abs(dn - np.median(dn))) + 1e-3)
    td = np.array(tdiff)
    cell_td = np.log(np.array(cell_td).mean(axis=0) + 1e-3)
    local = [
        float(np.log10(1.0 + np.percentile(sp_out, 90))),
        float(np.log10(1.0 + jump)),
        float(np.log10(np.mean(td < 0.25 * np.median(td)) + 0.01)),
        float(np.log10(td.max() / (np.median(td) + 1e-3) + 1e-3)),
        float(np.log10(1.0 + np.max(np.abs(_robust_z(cell_td))))),
        float(np.min(dups)) if dups else 0.0,
    ]
    return np.array(declared + measured + local, dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# Edge (pair) features: source/target content-consistency across a custody edge.
# ---------------------------------------------------------------------------------------------
PAIR_NAMES = ["psnr_mean", "psnr_std_t", "local_conc", "err_tjump", "log_frac_poor"]


def _gray_frames(path, n):
    c = av.open(str(path))
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    s.codec_context.thread_count = 2
    out = []
    for fr in c.decode(s):
        out.append(fr.to_ndarray(format="gray"))
        if len(out) >= n:
            break
    c.close()
    return out


def _pair_from_frames(src, tgt, gy=6, gx=8):
    """src, tgt: lists of equally sized gray frames (aligned at lag 0)."""
    n = min(len(src), len(tgt))
    h, w = tgt[0].shape
    mse, cell_conc = [], []
    for t in range(n):
        d = (src[t].astype(np.float32) - tgt[t].astype(np.float32)) ** 2
        mse.append(d.mean())
        e = d[: h // gy * gy, : w // gx * gx].reshape(gy, h // gy, gx, w // gx).mean(axis=(1, 3)).ravel()
        cell_conc.append(np.log10((e.max() + 1.0) / (np.median(e) + 1.0)))
    mse = np.array(mse)
    psnr = np.minimum(10 * np.log10(255.0 ** 2 / (mse + 1e-3)), 60.0)
    lm = np.log10(mse + 1.0)
    dl = np.diff(lm)
    jump = np.max(np.abs(dl - np.median(dl))) / (1.4826 * np.median(np.abs(dl - np.median(dl))) + 0.02)
    frac = np.mean(psnr < np.median(psnr) - 4.0)
    return np.array([float(psnr.mean()), float(psnr.std()), float(np.percentile(cell_conc, 90)),
                     float(np.log10(1.0 + jump)), float(np.log10(frac + 0.01))])


def pair_features_multi(src_path, tgt_paths, max_frames=180):
    """Pair features for several targets of the same source (source decoded once)."""
    src = _gray_frames(src_path, max_frames)
    cache = {}
    out = []
    for tp in tgt_paths:
        tgt = _gray_frames(tp, max_frames)
        size = (tgt[0].shape[1], tgt[0].shape[0])
        if size not in cache:
            cache[size] = [f if (f.shape[1], f.shape[0]) == size else cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in src]
        out.append(_pair_from_frames(cache[size], tgt))
    return out
