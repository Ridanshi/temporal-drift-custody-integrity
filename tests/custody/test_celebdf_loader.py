from pathlib import Path

import numpy as np
import pytest

from custody.celebdf_loader import (
    discover_pairs,
    real_heuristic_tamper_deltas,
    real_heuristic_tamper_joint_samples,
    reencoded_path_for,
)

CELEBDF_ROOT = Path(__file__).parent.parent.parent / "data" / "celebdf"
REENCODE_DIR = CELEBDF_ROOT / "manipulated_reencoded"


def test_discover_pairs_empty_when_root_missing():
    pairs = discover_pairs(str(Path("does") / "not" / "exist"))
    assert pairs == []


@pytest.mark.skipif(not CELEBDF_ROOT.exists(), reason="Celeb-DF dataset subset not downloaded")
def test_discover_pairs_finds_real_data():
    pairs = discover_pairs(str(CELEBDF_ROOT))
    assert len(pairs) > 0
    for p in pairs:
        assert Path(p["original"]).exists()
        assert Path(p["manipulated"]).exists()


@pytest.mark.skipif(not REENCODE_DIR.exists(), reason="Celeb-DF re-encoded subset not generated")
def test_reencoded_files_exist():
    pairs = discover_pairs(str(CELEBDF_ROOT))
    for p in pairs:
        out = reencoded_path_for(p["manipulated"], str(REENCODE_DIR))
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0


@pytest.mark.skipif(not REENCODE_DIR.exists(), reason="Celeb-DF re-encoded subset not generated")
def test_real_heuristic_tamper_deltas():
    deltas = real_heuristic_tamper_deltas(str(CELEBDF_ROOT), str(REENCODE_DIR))
    assert deltas.ndim == 2
    assert deltas.shape[0] > 0
    assert deltas.shape[1] == 4
    assert np.all(np.isfinite(deltas))


@pytest.mark.skipif(not REENCODE_DIR.exists(), reason="Celeb-DF re-encoded subset not generated")
def test_real_heuristic_tamper_joint_samples():
    samples = real_heuristic_tamper_joint_samples(str(CELEBDF_ROOT), str(REENCODE_DIR))
    assert samples.ndim == 2
    assert samples.shape[1] == 4
    assert np.all(np.isfinite(samples))
    pairs = discover_pairs(str(CELEBDF_ROOT))
    assert samples.shape[0] == len(pairs)
