import os
from pathlib import Path

import numpy as np

from custody.features import extract_intrinsic_features

REENCODE_SUFFIXES = {"whatsapp": "WA", "youtube": "YT"}


def _cache_path(root_dir: str) -> Path:
    return Path(root_dir) / ".feature_cache.npz"


def _features_for_paths(paths: list[str], root_dir: str) -> dict[str, np.ndarray]:
    cache_file = _cache_path(root_dir)
    cached: dict[str, np.ndarray] = {}
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        cached = data["features"].item()

    missing = [p for p in paths if p not in cached]
    if missing:
        for p in missing:
            cached[p] = extract_intrinsic_features(p)
        np.savez(cache_file, features=cached)

    return {p: cached[p] for p in paths}


VIDEO_EXTENSIONS = (".mp4", ".mov", ".3gp")


def _find_variant(root: Path, device_dir_name: str, scene: str, tag: str, device_code: str, stem: str) -> Path | None:
    scene_variant = f"{scene}{tag}"
    for ext in VIDEO_EXTENSIONS:
        candidate = root / device_dir_name / "videos" / scene_variant / f"{device_code}_V_{scene_variant}_{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def discover_triplets(root_dir: str) -> list[dict]:
    root = Path(root_dir)
    triplets = []
    if not root.exists():
        return triplets

    for device_dir in sorted(root.iterdir()):
        if not device_dir.is_dir():
            continue
        videos_dir = device_dir / "videos"
        if not videos_dir.exists():
            continue
        device_code = device_dir.name.split("_")[0]

        for scene_dir in sorted(videos_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene = scene_dir.name
            if scene.endswith("WA") or scene.endswith("YT"):
                continue

            # Native files may be .mp4 (Android) or .mov (Apple, native QuickTime
            # container); WA/YT re-uploads are typically transcoded to .mp4 by the
            # platform regardless of source format, so each side is matched independently.
            native_candidates = []
            prefix = f"{device_code}_V_{scene}_"
            for ext in VIDEO_EXTENSIONS:
                native_candidates.extend(scene_dir.glob(f"{prefix}*{ext}"))

            for native_path in sorted(native_candidates):
                stem = native_path.name[len(prefix) : -len(native_path.suffix)]

                wa_path = _find_variant(root, device_dir.name, scene, "WA", device_code, stem)
                yt_path = _find_variant(root, device_dir.name, scene, "YT", device_code, stem)
                if wa_path and yt_path:
                    triplets.append(
                        {
                            "device": device_dir.name,
                            "scene": scene,
                            "native": str(native_path),
                            "whatsapp": str(wa_path),
                            "youtube": str(yt_path),
                        }
                    )
    return triplets


def deltas_for_triplets(triplets: list[dict], root_dir: str, transform_type: str) -> np.ndarray:
    if transform_type not in REENCODE_SUFFIXES:
        raise ValueError(f"unknown transform_type: {transform_type}")
    key = "whatsapp" if transform_type == "whatsapp" else "youtube"

    all_paths = [t["native"] for t in triplets] + [t[key] for t in triplets]
    features = _features_for_paths(all_paths, root_dir)

    deltas = [features[t[key]] - features[t["native"]] for t in triplets]
    return np.array(deltas)


def joint_samples_for_triplets(triplets: list[dict], root_dir: str) -> np.ndarray:
    all_paths = []
    for t in triplets:
        all_paths.extend([t["native"], t["whatsapp"], t["youtube"]])
    features = _features_for_paths(all_paths, root_dir)

    return np.array([features[p] for p in all_paths])


def extract_genuine_deltas(root_dir: str, transform_type: str) -> np.ndarray:
    return deltas_for_triplets(discover_triplets(root_dir), root_dir, transform_type)


def extract_genuine_joint_samples(root_dir: str) -> np.ndarray:
    return joint_samples_for_triplets(discover_triplets(root_dir), root_dir)
