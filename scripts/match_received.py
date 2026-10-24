"""Match WhatsApp-received videos to the pilot clips and conditions by content.
  python scripts/match_received.py <received_dir> <out_map.csv>
Step 1 clip: MSE of the received file (first 180 frames, 160x90 gray) against each clip's sent native / ctrl / copymove.
Step 2 condition: native by length (>15 s); ctrl vs copymove by similarity inside the pixels where the sent copymove differs from the sent ctrl (frames 50-140)."""
import os, sys, csv, glob
import av, cv2, numpy as np
ROOT = "E:/DF paper/data/whatsapp_pilot"
SIZE = (160, 90)

def frames(f, n=180):
    c = av.open(f); out = []
    for i, fr in enumerate(c.decode(c.streams.video[0])):
        if i >= n: break
        out.append(cv2.resize(fr.to_ndarray(format="gray"), SIZE, interpolation=cv2.INTER_AREA).astype(np.float32))
    c.close(); return np.array(out)

def duration(f):
    c = av.open(f); d = float(c.duration / 1e6); c.close(); return d

rows = list(csv.DictReader(open(f"{ROOT}/manifest.csv")))
clips = {}
for r in rows:
    p = f"{ROOT}/Batch_{r['batch']}/{r['file']}"
    clips.setdefault(r["clip"], {})[r["condition"]] = p
want = [c for c in sorted(clips)]
sent = {}
for c in list(want):
    d = clips[c]
    try:
        sent[c] = {k: frames(d[k]) for k in ("native", "ctrl", "copymove") if k in d}
        sent[c]["mask"] = (np.abs(sent[c]["copymove"][50:140] - sent[c]["ctrl"][50:140]).mean(axis=0) > 6)
    except Exception as e:
        print("skip clip", c, repr(e)[:60]); want.remove(c); sent.pop(c, None)

def mse(a, b):
    m = min(len(a), len(b)); return float(((a[:m] - b[:m]) ** 2).mean())

out = []
for f in sorted(glob.glob(sys.argv[1] + "/*.mp4")):
    a = frames(f); dur = duration(f)
    sc = {c: min(mse(a, sent[c][k]) for k in ("native", "ctrl", "copymove")) for c in want if True}
    ranked = sorted(sc.items(), key=lambda t: t[1])
    clip, best = ranked[0]; second = ranked[1][1]
    kind = "native" if dur > 15 else None
    if kind is None:
        s = sent[clip]; mk = s["mask"]
        if mk.sum() >= 5 and len(a) > 140:
            dc = float(np.abs(a[50:140][:, mk] - s["ctrl"][50:140][:, mk]).mean()); dm = float(np.abs(a[50:140][:, mk] - s["copymove"][50:140][:, mk]).mean())
            kind = "copymove" if dm < dc else "ctrl"; margin = abs(dc - dm)
        else:
            kind = "ctrl?"; margin = 0.0; dc = dm = float("nan")
    else:
        margin = 0.0; dc = dm = float("nan")
    cc_ = av.open(f); sv = cc_.streams.video[0].codec_context; wh = f"{sv.width}x{sv.height}"; cc_.close()
    out.append((os.path.basename(f), clip, kind, round(best, 1), round(second, 1), round(margin, 2), round(dur, 1), round(dc, 3), round(dm, 3), wh))
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["file", "clip", "condition", "mse_best", "mse_second", "cond_margin", "dur", "d_ctrl", "d_copymove", "wh"]); w.writerows(out)
import collections
print(len(out), "files; per clip:", dict(collections.Counter(r[1] for r in out)))
for r in out: print(r)
