"""Real multi-hop re-encode chains. Hop 1 is the real platform file; later hops are real x264
re-encodes with platform specs (WhatsApp / YouTube), optionally with a tamper edit applied."""
import os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
import av, numpy as np
from custody.forge import render_intermediate, wa_target_dims
from custody.vision_loader import discover_triplets
sys.path.insert(0, "scripts")
from gen_forgeries import other_clip, WA_SPEC, T

YT_SPEC = dict(crf=25.0, preset="medium", profile="high", keyint=90, bframes=2, scenecut=40)
SCR = "C:/Users/Ridan/AppData/Local/Temp/claude/E--DF-paper/4ac6b769-ad6d-461e-b926-3277c9e8c3af/scratchpad/chain"
OUT = "E:/DF paper/work/chains"
os.makedirs(SCR, exist_ok=True); os.makedirs(OUT, exist_ok=True)
SPEC = {"WA": WA_SPEC, "YT": YT_SPEC}
ROUTE = {"WAfirst": ["WA", "YT", "WA", "YT"], "YTfirst": ["YT", "WA", "YT", "WA"]}


def dims(path):
    c = av.open(path); cc = c.streams.video[0].codec_context; w, h = cc.width, cc.height; c.close()
    return w, h


def hop(prev, out, plat, op=None, scale=1.0, seed=0, other=None):
    w, h = dims(prev)
    tgt = wa_target_dims(w, h) if plat == "WA" else (w, h)
    render_intermediate(prev, out, op, seed=seed, other=other, target=tgt, tamper_scale=scale, direct=SPEC[plat])
    return out


def job(args):
    for att in range(5):
        try:
            return _job(args)
        except Exception as e:
            print('retry', args, att, repr(e)[:80], flush=True)
            time.sleep(20)
    raise RuntimeError(f'failed {args}')


def _job(args):
    scen, route, i, op = args
    t = T[i]
    plats = ROUTE[route]
    first = t["whatsapp"] if plats[0] == "WA" else t["youtube"]
    prev = first
    tag = f"{scen}_{route}_{i:03d}_{op or 'none'}"
    for k in range(1, 4):
        out = f"{SCR}/{threading.get_ident()}_{k}.mp4"
        o, sc = None, 1.0
        if scen == "tamper2" and k == 1:
            o, sc = op, 1.0
        elif scen == "creep":
            o, sc = op, 0.34
        hop(prev, out, plats[k], o, sc, seed=1000 + i, other=other_clip(i))
        keep = f"{OUT}/{tag}_h{k + 1}.mp4"
        import shutil; shutil.copyfile(out, keep)
        prev = keep
    return (scen, route, i, op), True


if __name__ == "__main__":
    ops3 = ("copymove", "blur", "recolor")
    jobs = [("genuine", r, i, None) for r in ROUTE for i in range(len(T))]
    jobs += [("tamper2", "WAfirst", i, op) for i in range(0, len(T), 2) for op in ops3]
    jobs += [("creep", "WAfirst", i, op) for i in range(0, len(T), 2) for op in ops3]
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    if only:
        jobs = [j for j in jobs if j[0] in only]
    cache = "work/chain_done.npz"
    done = np.load(cache, allow_pickle=True)["features"].item() if os.path.exists(cache) else {}
    todo = [j for j in jobs if j not in done]
    print("todo", len(todo), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(int(os.environ.get('WORKERS', 2))) as ex:
        for n, (k, f) in enumerate(ex.map(job, todo)):
            done[k] = f
            if n % 20 == 19:
                np.savez(cache, features=np.array(done, dtype=object)); print(n + 1, round(time.time() - t0), "s", flush=True)
    np.savez(cache, features=np.array(done, dtype=object)); print("finished", len(done), round(time.time() - t0), "s")
