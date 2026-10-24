"""Prepare the real-WhatsApp pilot package from the FloreView clips (data/floreview).

Per device, four files to send through WhatsApp as VIDEOS (not documents):
  native   - the original recording, unmodified            (genuine transfer of a native clip)
  ctrl     - untampered re-encode of its first 180 frames   (same pre-encode as the forgeries, no edit)
  copymove - copy-move edit in frames 50-140, same pre-encode
  blur     - local Gaussian blur in frames 50-140, same pre-encode
Output: data/whatsapp_pilot/Batch_N/ + manifest.csv + SEND_THESE_IN_ORDER.txt
"""
import os, sys, csv, shutil, glob, json, time
import av, cv2
from concurrent.futures import ThreadPoolExecutor
import custody.forge as F

SRC = "E:/DF paper/data/floreview"
OUT = "E:/DF paper/data/whatsapp_pilot"
HQ = dict(crf=14.0, preset="slow", profile="high", keyint=60, bframes=0)
EDITS = ("copymove", "blur")
PER_BATCH = 15


def clips():
    out = []
    for d in sorted(os.listdir(SRC)):
        fs = [f for f in glob.glob(f"{SRC}/{d}/*") if f.lower().endswith((".mp4", ".mov", ".3gp"))]
        if fs: out.append((d, fs[0]))
    return out


def dims(path):
    first = next(F._frames(path, 1)); H, W = first.shape[:2]
    s = min(1.0, 1920.0 / max(H, W))
    return (int(W * s) // 2 * 2, int(H * s) // 2 * 2), (W, H)


def other_native(i, cl):
    j = (i + 1) % len(cl)
    return cl[j][1]


def job(args):
    i, dev, path, batch = args
    tag = f"FV{i + 1:02d}"
    od = f"{OUT}/Batch_{batch}"; os.makedirs(od, exist_ok=True)
    rows = []
    ext = os.path.splitext(path)[1].lower()
    nat = f"{od}/{tag}_native{ext}"
    if not os.path.exists(nat): shutil.copy2(path, nat)
    rows.append((tag, dev, "native", os.path.basename(nat)))
    target, _ = dims(path)
    for cond, op in (("ctrl", None),) + tuple((e, e) for e in EDITS):
        p = f"{od}/{tag}_{cond}.mp4"
        if not os.path.exists(p):
            F.render_intermediate(path, p, op, seed=7000 + i, other=other_native(i, CL), target=target, direct=HQ)
        rows.append((tag, dev, cond, os.path.basename(p)))
    return rows


CL = clips()
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(i, d, p, i // PER_BATCH + 1) for i, (d, p) in enumerate(CL)]
    print(len(jobs), "clips,", (len(jobs) - 1) // PER_BATCH + 1, "batches")
    t0 = time.time(); manifest = []
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 2))) as ex:
        for r in ex.map(job, jobs):
            manifest += r
    with open(f"{OUT}/manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "device", "condition", "file", "batch", "size_MB"])
        for tag, dev, cond, fn in manifest:
            b = int(tag[2:]) - 1; b = b // PER_BATCH + 1
            w.writerow([tag, dev, cond, fn, b, round(os.path.getsize(f"{OUT}/Batch_{b}/{fn}") / 1e6, 1)])
    print("prepared", len(manifest), "files in %.0fs" % (time.time() - t0))
