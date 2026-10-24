"""Turn the matcher's output into the scoring map, keeping only unambiguous, genuinely recompressed files.
  python scripts/build_pilot_map.py work/real_match_zip1.csv work/real_pilot_map2.csv"""
import csv, sys, collections, math
ROOT = "E:/DF paper/data/whatsapp_pilot"
Z = f"{ROOT}/received/zip_1"
man = {}
for r in csv.DictReader(open(f"{ROOT}/manifest.csv")):
    man.setdefault(r["clip"], {})[r["condition"]] = f"{ROOT}/Batch_{r['batch']}/{r['file']}"
rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.defaultdict(list)
for r in rows:
    r["mse_best"] = float(r["mse_best"]); r["mse_second"] = float(r["mse_second"]); r["dur"] = float(r["dur"])
    r["dc"] = float(r["d_ctrl"]) if r["d_ctrl"] not in ("", "nan") else math.nan
    r["dm"] = float(r["d_copymove"]) if r["d_copymove"] not in ("", "nan") else math.nan
    by[r["clip"]].append(r)
out, log = [], []
DROP = {"FV01": "covered by the first lot (received/Batch_1)", "FV15": "ambiguous (two natives, 3.0 s clips)", "FV02": "conditions undetermined"}
for clip in sorted(by):
    if clip in DROP: log.append((clip, "drop: " + DROP[clip])); continue
    fs = by[clip]; src = man[clip]["native"]
    ok = [r for r in fs if not r["wh"].startswith("1920") and r["mse_best"] >= 0.5 and r["mse_second"] >= 1.3 * r["mse_best"]]
    nat = [r for r in ok if r["dur"] > 15]; six = [r for r in ok if r["dur"] <= 15]
    notes = []
    if len(nat) == 1: out.append((clip, "native", src, f"{Z}/{nat[0]['file']}"))
    elif len(nat) > 1: notes.append("2 natives -> native dropped")
    if len(fs) - len(ok): notes.append(f"{len(fs) - len(ok)} file(s) rejected (not recompressed / low match)")
    if len(six) == 1 and not math.isnan(six[0]["dm"]):
        out.append((clip, "copymove" if six[0]["dm"] < six[0]["dc"] else "ctrl", src, f"{Z}/{six[0]['file']}"))
    elif len(six) == 2 and not any(math.isnan(r["dm"]) for r in six):
        a, b = sorted(six, key=lambda r: r["dm"] - r["dc"])
        out += [(clip, "copymove", src, f"{Z}/{a['file']}"), (clip, "ctrl", src, f"{Z}/{b['file']}")]
        notes.append("two 6 s files: lower (dm-dc) = copymove")
    elif len(six) >= 3: notes.append("3+ short files -> ambiguous, dropped")
    log.append((clip, "; ".join(notes) or "ok"))
# FV01 from the first lot
f1 = f"{ROOT}/received/Batch_1/WhatsApp Video 2026-10-24 at 10.33."
for cond, ts in (("native", "05"), ("ctrl", "06"), ("copymove", "11"), ("blur", "12")):
    out.append(("FV01", cond, man["FV01"]["native"], f"{f1}{ts} PM.mp4"))
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip", "condition", "source", "received"]); w.writerows(out)
cnt = collections.defaultdict(dict)
for c, cond, *_ in out: cnt[c][cond] = 1
print(len(out), "usable files over", len(cnt), "clips")
print("clips with ctrl+copymove (edit-only usable):", sum(1 for c in cnt if "ctrl" in cnt[c] and "copymove" in cnt[c]))
print("clips with native:", sum(1 for c in cnt if "native" in cnt[c]))
for c, n in log: print(" ", c, n)
