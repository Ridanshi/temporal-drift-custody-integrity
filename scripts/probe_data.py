import json
import sys
from pathlib import Path

import av

from custody.vision_loader import discover_triplets

root = sys.argv[1] if len(sys.argv) > 1 else "E:/DF paper/data/vision"
rows = []
for t in discover_triplets(root):
    for kind in ("native", "whatsapp", "youtube"):
        p = t[kind]
        c = av.open(p)
        s = c.streams.video[0]
        cc = s.codec_context
        rows.append(dict(device=t["device"], scene=t["scene"], kind=kind, codec=cc.name,
            profile=str(cc.profile), w=cc.width, h=cc.height,
            fps=float(s.average_rate) if s.average_rate else None, nframes=s.frames,
            dur=float(c.duration / 1e6) if c.duration else None, bitrate=c.bit_rate,
            mb=round(Path(p).stat().st_size / 1e6, 1), pix=str(cc.pix_fmt)))
        c.close()
json.dump(rows, open("scripts/probe_vision.json", "w"), indent=1)
print(len(rows))
