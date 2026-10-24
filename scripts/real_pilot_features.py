"""Features (file, source-vs-copy pair, local) for the usable real-WhatsApp pilot files. Resumable.
  python scripts/real_pilot_features.py <map.csv> <out.pkl>"""
import os, sys, csv, pickle, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import cv2; cv2.setNumThreads(1)
from custody.features_v2 import extract_features_ext, pair_features_multi
from custody.local_features import decode_small, local_pair_features
rows = list(csv.DictReader(open(sys.argv[1])))
done = pickle.load(open(sys.argv[2], "rb")) if os.path.exists(sys.argv[2]) and os.path.getsize(sys.argv[2]) else {}
by = {}
for r in rows: by.setdefault(r["source"], []).append(r)
t0 = time.time()
for src, rs in by.items():
    todo = [r for r in rs if (r["clip"], r["condition"]) not in done]
    if not todo: continue
    try:
        files = [r["received"] for r in todo]
        pairs = pair_features_multi(src, files); sm = decode_small(src); se = extract_features_ext(src)
        for r, pv in zip(todo, pairs):
            done[(r["clip"], r["condition"])] = dict(ext=extract_features_ext(r["received"]), pair=pv, loc=local_pair_features(sm, decode_small(r["received"])), src_ext=se, received=r["received"])
    except Exception as e:
        print("FAIL", src[-30:], repr(e)[:100], flush=True)
    pickle.dump(done, open(sys.argv[2] + ".tmp", "wb")); os.replace(sys.argv[2] + ".tmp", sys.argv[2])
    print(len(done), "done", round(time.time() - t0), "s", flush=True)
print("finished", len(done))
