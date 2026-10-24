"""Download one short Flat H.264 clip per device from FloreView (CC BY-SA 4.0, https://lesc.dinfo.unifi.it/FloreView/).
Reads the selection written during the dataset search; resumable; verifies size."""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

SEL = sys.argv[1]
OUT = "E:/DF paper/data/floreview"

def get(item):
    dev, url, size = item
    d = f"{OUT}/{dev}"; os.makedirs(d, exist_ok=True)
    p = f"{d}/{url.rsplit('/', 1)[-1]}"
    if os.path.exists(p) and os.path.getsize(p) == size:
        return dev, "have"
    for a in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(p + ".part", "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b)
            if os.path.getsize(p + ".part") != size: raise IOError("size mismatch")
            os.replace(p + ".part", p); return dev, "ok"
        except Exception as e:
            time.sleep(5)
    return dev, "FAILED"

if __name__ == "__main__":
    sel = json.load(open(SEL)); t0 = time.time()
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(get, sel))
    print({k: sum(1 for _, s in res if s == k) for k in ("ok", "have", "FAILED")}, "%.0fs" % (time.time() - t0))
    print([d for d, s in res if s == "FAILED"])
