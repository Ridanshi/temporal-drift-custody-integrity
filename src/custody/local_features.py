"""Windowed, region-wise source-versus-copy features.

Both videos are decoded to a fixed 240x424 grayscale canvas. For each 30-frame window and each cell of a
6x8 grid we measure error, sharpness change, residual-noise correlation and motion change between the
source and the copy, then report how far the worst windows and cells stand out from the rest of the same clip.
"""
import av
import cv2
import numpy as np

H, W, GY, GX, WIN_FRAMES = 240, 424, 6, 8, 30
NAMES = ["loc_err_max", "loc_err_top", "loc_sharp_max", "loc_sharp_top", "loc_noise_max", "loc_noise_top", "loc_motion_max", "loc_motion_top"]


def decode_small(path, n=180):
    c = av.open(str(path))
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    s.codec_context.thread_count = 2
    out = []
    for fr in c.decode(s):
        g = fr.to_ndarray(format="gray")
        out.append(cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA))
        if len(out) >= n:
            break
    c.close()
    return np.array(out, dtype=np.uint8)


def _cells(a):
    return a.reshape(GY, H // GY, GX, W // GX)


def _cell_mean(a):
    return _cells(a).mean(axis=(1, 3))


def _cell_sum(a):
    return _cells(a).sum(axis=(1, 3))


def _robust_z(x, floor):
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    return (x - med) / (mad + floor)


def local_pair_features(src, tgt):
    n = min(len(src), len(tgt))
    nw = n // WIN_FRAMES
    if nw < 2:
        raise ValueError("too few frames")
    mse = np.zeros((nw, GY, GX)); lap_s = np.zeros_like(mse); lap_t = np.zeros_like(mse)
    ncc_num = np.zeros_like(mse); ncc_a = np.zeros_like(mse); ncc_b = np.zeros_like(mse)
    td_s = np.zeros_like(mse); td_t = np.zeros_like(mse)
    for t in range(nw * WIN_FRAMES):
        w = t // WIN_FRAMES
        s = src[t]; g = tgt[t]
        sf = s.astype(np.float32); gf = g.astype(np.float32)
        mse[w] += _cell_mean((sf - gf) ** 2)
        lap_s[w] += _cell_mean(cv2.Laplacian(sf, cv2.CV_32F) ** 2)
        lap_t[w] += _cell_mean(cv2.Laplacian(gf, cv2.CV_32F) ** 2)
        rs = sf - cv2.medianBlur(s, 3).astype(np.float32)
        rt = gf - cv2.medianBlur(g, 3).astype(np.float32)
        ncc_num[w] += _cell_sum(rs * rt); ncc_a[w] += _cell_sum(rs * rs); ncc_b[w] += _cell_sum(rt * rt)
        if t > 0:
            td_s[w] += _cell_mean(np.abs(sf - src[t - 1].astype(np.float32)))
            td_t[w] += _cell_mean(np.abs(gf - tgt[t - 1].astype(np.float32)))
    k = WIN_FRAMES
    m_err = np.log10(mse / k + 1.0)
    m_sharp = np.abs(np.log((lap_t / k + 1.0) / (lap_s / k + 1.0)))
    m_noise = 1.0 - ncc_num / (np.sqrt(ncc_a * ncc_b) + 1e-6)
    m_motion = np.abs(np.log((td_t / k + 0.5) / (td_s / k + 0.5)))
    out = []
    for m, fl in ((m_err, 0.05), (m_sharp, 0.05), (m_noise, 0.03), (m_motion, 0.05)):
        z = _robust_z(m.ravel(), fl)
        zs = np.sort(z)[::-1]
        out += [float(np.log1p(max(zs[0], 0.0))), float(np.log1p(max(zs[:10].mean(), 0.0)))]
    return np.array(out)
