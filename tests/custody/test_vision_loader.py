from pathlib import Path

import numpy as np
import pytest

from custody.vision_loader import (
    deltas_for_triplets,
    discover_triplets,
    extract_genuine_deltas,
    extract_genuine_joint_samples,
    joint_samples_for_triplets,
)

VISION_ROOT = Path(__file__).parent.parent.parent / "data" / "vision"


def test_discover_triplets_empty_when_root_missing():
    triplets = discover_triplets(str(Path("does") / "not" / "exist"))
    assert triplets == []


@pytest.mark.skipif(not VISION_ROOT.exists(), reason="VISION dataset subset not downloaded")
def test_discover_triplets_finds_real_data():
    triplets = discover_triplets(str(VISION_ROOT))
    assert len(triplets) > 0
    for t in triplets:
        assert Path(t["native"]).exists()
        assert Path(t["whatsapp"]).exists()
        assert Path(t["youtube"]).exists()


@pytest.mark.skipif(not VISION_ROOT.exists(), reason="VISION dataset subset not downloaded")
def test_extract_genuine_deltas_whatsapp():
    deltas = extract_genuine_deltas(str(VISION_ROOT), "whatsapp")
    assert deltas.ndim == 2
    assert deltas.shape[0] > 0
    assert deltas.shape[1] == 4
    assert np.all(np.isfinite(deltas))


@pytest.mark.skipif(not VISION_ROOT.exists(), reason="VISION dataset subset not downloaded")
def test_extract_genuine_joint_samples():
    samples = extract_genuine_joint_samples(str(VISION_ROOT))
    assert samples.ndim == 2
    assert samples.shape[1] == 4
    assert np.all(np.isfinite(samples))


@pytest.mark.skipif(not VISION_ROOT.exists(), reason="VISION dataset subset not downloaded")
def test_deltas_for_triplets_matches_full_extraction_on_full_set():
    triplets = discover_triplets(str(VISION_ROOT))
    subset_deltas = deltas_for_triplets(triplets, str(VISION_ROOT), "whatsapp")
    full_deltas = extract_genuine_deltas(str(VISION_ROOT), "whatsapp")
    assert np.allclose(subset_deltas, full_deltas)


@pytest.mark.skipif(not VISION_ROOT.exists(), reason="VISION dataset subset not downloaded")
def test_deltas_for_triplets_respects_subset():
    triplets = discover_triplets(str(VISION_ROOT))
    devices = sorted({t["device"] for t in triplets})
    held_out = [t for t in triplets if t["device"] == devices[0]]
    rest = [t for t in triplets if t["device"] != devices[0]]

    held_out_deltas = deltas_for_triplets(held_out, str(VISION_ROOT), "whatsapp")
    rest_deltas = deltas_for_triplets(rest, str(VISION_ROOT), "whatsapp")

    assert held_out_deltas.shape[0] == len(held_out)
    assert rest_deltas.shape[0] == len(rest)
    assert held_out_deltas.shape[0] + rest_deltas.shape[0] == len(triplets)


@pytest.mark.skipif(not VISION_ROOT.exists(), reason="VISION dataset subset not downloaded")
def test_joint_samples_for_triplets_respects_subset():
    triplets = discover_triplets(str(VISION_ROOT))
    devices = sorted({t["device"] for t in triplets})
    held_out = [t for t in triplets if t["device"] == devices[0]]

    samples = joint_samples_for_triplets(held_out, str(VISION_ROOT))
    assert samples.shape == (len(held_out) * 3, 4)
