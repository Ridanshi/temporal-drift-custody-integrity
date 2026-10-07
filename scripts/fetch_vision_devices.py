"""Recursively downloads VISION dataset device video folders from the Apache-style
directory listing, into the exact structure custody.vision_loader expects.
Skips files that already exist so it's safe to re-run / resume.
"""
import os
import re
import sys
import time

import requests

BASE = "https://lesc.dinfo.unifi.it/VISION/dataset"
OUT_ROOT = r"D:\DF paper\data\vision"
LINK_RE = re.compile(r'href="([^"?][^"]*)"')

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})


def list_dir(url: str) -> list[str]:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    hrefs = LINK_RE.findall(resp.text)
    return [h for h in hrefs if h != "../"]


def download_file(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  skip (exists): {os.path.basename(dest)}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                written += len(chunk)
        os.replace(tmp, dest)
    print(f"  downloaded: {os.path.basename(dest)} ({written / 1e6:.1f} MB)")


def download_videos_tree(device_folder: str) -> None:
    videos_url = f"{BASE}/{device_folder}/videos/"
    out_videos_dir = os.path.join(OUT_ROOT, device_folder, "videos")
    print(f"=== {device_folder} ===")
    try:
        scene_entries = list_dir(videos_url)
    except Exception as e:
        print(f"  FAILED to list {videos_url}: {e}")
        return

    for entry in scene_entries:
        if not entry.endswith("/"):
            continue
        scene = entry.rstrip("/")
        scene_url = videos_url + entry
        try:
            files = list_dir(scene_url)
        except Exception as e:
            print(f"  FAILED to list {scene_url}: {e}")
            continue
        for fname in files:
            if fname.endswith("/"):
                continue
            if not fname.lower().endswith((".mp4", ".mov")):
                continue
            file_url = scene_url + fname
            dest = os.path.join(out_videos_dir, scene, fname)
            for attempt in range(3):
                try:
                    download_file(file_url, dest)
                    break
                except Exception as e:
                    print(f"  retry {attempt+1}/3 for {fname}: {e}")
                    time.sleep(2)


if __name__ == "__main__":
    devices = sys.argv[1:]
    if not devices:
        print("usage: fetch_vision_devices.py <DeviceFolder1> [DeviceFolder2] ...")
        sys.exit(1)
    for d in devices:
        download_videos_tree(d)
    print("ALL DONE")
