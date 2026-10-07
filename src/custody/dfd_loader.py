import json
from pathlib import Path

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


def real_tamper_deltas(root_dir: str) -> np.ndarray:
    """Manipulated-minus-original intrinsic-feature deltas for real DFD face-swap pairs.

    This is a real Tier-1 naive-tamper instance: manipulated content with no attempt to
    make its intrinsic drift consistent with any declared custody transform, exactly the
    Section 3 Tier-1 definition, just drawn from real deepfake video instead of a
    statistically simulated one.
    """
    pairs = discover_pairs(root_dir)
    all_paths = [p["original"] for p in pairs] + [p["manipulated"] for p in pairs]
    features = _features_for_paths(all_paths, root_dir)

    deltas = [features[p["manipulated"]] - features[p["original"]] for p in pairs]
    return np.array(deltas)


def real_tamper_joint_samples(root_dir: str) -> np.ndarray:
    """Intrinsic feature vectors of the manipulated clips, for scoring against the
    co-evasion test's genuine-state joint distribution."""
    pairs = discover_pairs(root_dir)
    all_paths = [p["manipulated"] for p in pairs]
    features = _features_for_paths(all_paths, root_dir)
    return np.array([features[p] for p in all_paths])
