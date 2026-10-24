"""Platform-aware forgeries with weakened edits (tamper strength 0.25 / 0.5), every other clip."""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
import av
from custody.forge import render_intermediate, wa_target_dims, OPS
sys.path.insert(0, "scripts")
from gen_forgeries import T, WA_SPEC, other_clip

OUT = "E:/DF paper/work/forged"
SCALES = {"s25": 0.25, "s50": 0.5}


def job(a):
    tier, i, op = a
    t = T[i]
    c = av.open(t["native"]); cc = c.streams.video[0].codec_context; w, h = cc.width, cc.height; c.close()
    os.makedirs(f"{OUT}/{tier}", exist_ok=True)
    out = f"{OUT}/{tier}/{i:03d}_{op}.mp4"
    if os.path.exists(out):
        return a
    for att in range(3):
        try:
            render_intermediate(t["native"], out, op, seed=1000 + i, other=other_clip(i), target=wa_target_dims(w, h),
                                tamper_scale=SCALES[tier], direct=WA_SPEC)
            return a
        except Exception as e:
            time.sleep(8)
    raise RuntimeError(a)


if __name__ == "__main__":
    jobs = [(tier, i, op) for tier in SCALES for i in range(0, len(T), 2) for op in OPS]
    t0 = time.time()
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 3))) as ex:
        for n, _ in enumerate(ex.map(job, jobs)):
            if n % 50 == 49:
                print(n + 1, round(time.time() - t0), "s", flush=True)
    print("finished", round(time.time() - t0), "s")
