import json
from pathlib import Path

import av
import numpy as np

from custody.features import extract_intrinsic_features


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


def discover_pairs(root_dir: str) -> list[dict]:
    manifest_path = Path(root_dir) / "manifest.json"
    if not manifest_path.exists():
        return []
    with open(manifest_path) as f:
        pairs = json.load(f)
    return pairs


def heuristic_reencode(input_path: str, output_path: str, crf: int = 28, preset: str = "medium") -> None:
    """Off-the-shelf standard re-encode: a fixed CRF/preset with no knowledge of any
    detector, exactly the Section 3 Tier-2 definition (standard re-encode presets, no
    adaptation to the detector under test)."""
    if Path(output_path).exists():
        return
    in_container = av.open(input_path)
    in_stream = in_container.streams.video[0]

    out_container = av.open(output_path, mode="w")
    out_stream = out_container.add_stream("libx264", rate=in_stream.average_rate)
    out_stream.width = in_stream.width
    out_stream.height = in_stream.height
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": str(crf), "preset": preset}

    for frame in in_container.decode(in_stream):
        frame = frame.reformat(format="yuv420p")
        for packet in out_stream.encode(frame):
            out_container.mux(packet)
    for packet in out_stream.encode():
        out_container.mux(packet)

    out_container.close()
    in_container.close()


def reencoded_path_for(manipulated_path: str, reencode_dir: str) -> str:
    return str(Path(reencode_dir) / Path(manipulated_path).name)


def real_heuristic_tamper_deltas(root_dir: str, reencode_dir: str) -> np.ndarray:
    """Manipulated-and-off-the-shelf-re-encoded minus original intrinsic-feature deltas:
    a real Tier-2 heuristic-co-evasion instance (Section 3), real deepfake content plus a
    real standard re-encode attempting to look like a normal transform, with no adaptation
    to this detector's thresholds."""
    pairs = discover_pairs(root_dir)
    reencoded_paths = [reencoded_path_for(p["manipulated"], reencode_dir) for p in pairs]
    all_paths = [p["original"] for p in pairs] + reencoded_paths
    features = _features_for_paths(all_paths, root_dir)

    deltas = [
        features[reencoded_path_for(p["manipulated"], reencode_dir)] - features[p["original"]]
        for p in pairs
    ]
    return np.array(deltas)


def real_heuristic_tamper_joint_samples(root_dir: str, reencode_dir: str) -> np.ndarray:
    pairs = discover_pairs(root_dir)
    reencoded_paths = [reencoded_path_for(p["manipulated"], reencode_dir) for p in pairs]
    features = _features_for_paths(reencoded_paths, root_dir)
    return np.array([features[p] for p in reencoded_paths])
