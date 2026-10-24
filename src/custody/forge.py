"""Real forgery generation: tamper decoded frames, then re-encode with an x264 pipeline.

An attacker tampers with a native recording and re-encodes it so that it can be passed
off as the output of a platform transfer (WhatsApp, YouTube).  Nothing here simulates
feature values; every forged file is a real H.264 file produced by ffmpeg.
"""
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import cv2
import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
OPS = ("copymove", "blur", "splice", "recolor", "dup")
EXTRA_OPS = ("text", "inpaint", "cut")
WIN = (50, 140)  # tampered frame window inside the first MAX_FRAMES frames
MAX_FRAMES = 180


def wa_target_dims(w, h):
    """Output size of the WhatsApp transfer as observed on the VISION clips."""
    return (848, 480) if w / h >= 1.7 else (w, h)


def _rect(rng, W, H, fw, fh, left=None):
    rw, rh = int(W * fw) // 2 * 2, int(H * fh) // 2 * 2
    x = int(rng.integers(0, W - rw)) if left is None else int(rng.integers(0, max(1, W // 2 - rw))) + (0 if left else W // 2)
    y = int(rng.integers(0, H - rh))
    return x, y, rw, rh


def _frames(path, n):
    c = av.open(str(path))
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    s.codec_context.thread_count = 2
    out = 0
    for fr in c.decode(s):
        yield fr.to_ndarray(format="bgr24")
        out += 1
        if out >= n:
            break
    c.close()


def _fps(path):
    c = av.open(str(path))
    r = c.streams.video[0].average_rate
    c.close()
    return Fraction(r).limit_denominator(1000)


def render_intermediate(native, out_path, op=None, seed=0, other=None, target=None, tamper_scale=1.0, direct=None):
    """Decode `native`, apply tamper `op` in WIN, scale to `target`, write lossless FFV1
    (or, with `direct`, encode straight to H.264 with those x264 settings).

    tamper_scale in (0,1] scales the strength of the edit (used for creeping chains).
    """
    rng = np.random.default_rng(seed)
    fps = _fps(native)
    first = next(_frames(native, 1))
    H, W = first.shape[:2]
    tw, th = target or (W, H)
    tw, th = tw // 2 * 2, th // 2 * 2
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{tw}x{th}",
           "-r", str(float(fps)), "-i", "-"] + (x264_args(**direct) if direct else ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p"]) + [str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    src = _rect(rng, W, H, 0.22, 0.22, left=True)
    dst = _rect(rng, W, H, 0.22, 0.22, left=False)
    reg = _rect(rng, W, H, 0.30, 0.30)
    reg_big = _rect(rng, W, H, 0.5, 0.5)
    oth = _frames(other, MAX_FRAMES) if (op == "splice" and other) else None
    held = None
    for i, fr in enumerate(_frames(native, MAX_FRAMES)):
        ofr = next(oth, None) if oth is not None else None
        if op == "cut" and 70 <= i < 100:
            continue
        if op and WIN[0] <= i < WIN[1]:
            if op == "text":
                x, y, rw, rh = reg
                cv2.putText(fr, "CONFIDENTIAL", (x, y + rh // 2), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, W / 900.0), (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(fr, "CONFIDENTIAL", (x, y + rh // 2), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, W / 900.0), (255, 255, 255), 2, cv2.LINE_AA)
            elif op == "inpaint":
                x, y, rw, rh = src
                mh, mw = rh // 2, rw // 2
                y0_, x0_ = max(0, y - 16), max(0, x - 16)
                crop = fr[y0_:y + mh + 16, x0_:x + mw + 16].copy()
                m = np.zeros(crop.shape[:2], np.uint8); m[y - y0_:y - y0_ + mh, x - x0_:x - x0_ + mw] = 255
                fr[y0_:y + mh + 16, x0_:x + mw + 16] = cv2.inpaint(crop, m, 3, cv2.INPAINT_TELEA)
            if op == "copymove":
                sx, sy, rw, rh = src
                dx, dy, _, _ = dst
                patch = fr[sy:sy + rh, sx:sx + rw].copy()
                fr[dy:dy + rh, dx:dx + rw] = cv2.addWeighted(fr[dy:dy + rh, dx:dx + rw], 1 - tamper_scale, patch, tamper_scale, 0)
            elif op == "blur":
                x, y, rw, rh = reg
                sigma = 1.0 + 4.0 * tamper_scale
                fr[y:y + rh, x:x + rw] = cv2.GaussianBlur(fr[y:y + rh, x:x + rw], (0, 0), sigma)
            elif op == "splice" and ofr is not None:
                x, y, rw, rh = reg
                o = cv2.resize(ofr, (W, H))[y:y + rh, x:x + rw]
                fr[y:y + rh, x:x + rw] = cv2.addWeighted(fr[y:y + rh, x:x + rw], 1 - tamper_scale, o, tamper_scale, 0)
            elif op == "recolor":
                x, y, rw, rh = reg_big
                reg_px = fr[y:y + rh, x:x + rw].astype(np.float32)
                a = 1.0 + 0.25 * tamper_scale
                b = -12.0 * tamper_scale
                fr[y:y + rh, x:x + rw] = np.clip(reg_px * a + b, 0, 255).astype(np.uint8)
            elif op == "dup":
                if held is None:
                    held = fr.copy()
                fr = held if tamper_scale >= 1.0 else cv2.addWeighted(fr, 1 - tamper_scale, held, tamper_scale, 0)
        if (tw, th) != (W, H):
            fr = cv2.resize(fr, (tw, th), interpolation=cv2.INTER_AREA)
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg intermediate failed")
    return out_path


def x264_args(crf=28.0, preset="medium", profile="baseline", keyint=30, bframes=0, deblock=(0, 0),
              pre_noise=0.0, denoise=0.0, maxrate_kbps=None, threads=2, tune=None, scenecut=0, refs=None):
    vf = []
    if denoise > 0:
        vf.append(f"hqdn3d={denoise}:{denoise}:{denoise}:{denoise}")
    if pre_noise > 0:
        vf.append(f"noise=alls={int(pre_noise)}:allf=t")
    xp = f"keyint={int(keyint)}:min-keyint={int(min(keyint, 25))}:bframes={int(bframes)}:deblock={deblock[0]}:{deblock[1]}:scenecut={scenecut}"
    if refs:
        xp += f":ref={int(refs)}"
    args = ["-threads", str(threads)]
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-c:v", "libx264", "-preset", preset, "-profile:v", profile, "-crf", f"{crf:.2f}", "-x264-params", xp,
             "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart"]
    if maxrate_kbps:
        args += ["-maxrate", f"{int(maxrate_kbps)}k", "-bufsize", f"{int(2 * maxrate_kbps)}k"]
    if tune:
        args += ["-tune", tune]
    return args


def encode(inter, out_path, **kw):
    """x264 encode of a lossless intermediate; returns out_path."""
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(inter)] + x264_args(**kw) + [str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    return out_path
