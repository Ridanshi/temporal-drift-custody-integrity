from pathlib import Path

import numpy as np
import pytest

from custody.dfd_loader import discover_pairs, real_tamper_deltas, real_tamper_joint_samples

DFD_ROOT = Path(__file__).parent.parent.parent / "data" / "dfd"


def test_discover_pairs_empty_when_root_missing():
    pairs = discover_pairs(str(Path("does") / "not" / "exist"))
    assert pairs == []


@pytest.mark.skipif(not DFD_ROOT.exists(), reason="DFD dataset subset not downloaded")
def test_discover_pairs_finds_real_data():
    pairs = discover_pairs(str(DFD_ROOT))
    assert len(pairs) > 0
    for p in pairs:
        assert Path(p["original"]).exists()
        assert Path(p["manipulated"]).exists()


@pytest.mark.skipif(not DFD_ROOT.exists(), reason="DFD dataset subset not downloaded")
def test_real_tamper_deltas():
    deltas = real_tamper_deltas(str(DFD_ROOT))
    assert deltas.ndim == 2
    assert deltas.shape[0] > 0
    assert deltas.shape[1] == 4
    assert np.all(np.isfinite(deltas))


@pytest.mark.skipif(not DFD_ROOT.exists(), reason="DFD dataset subset not downloaded")
def test_real_tamper_joint_samples():
    samples = real_tamper_joint_samples(str(DFD_ROOT))
    assert samples.ndim == 2
    assert samples.shape[1] == 4
    assert np.all(np.isfinite(samples))
    pairs = discover_pairs(str(DFD_ROOT))
    assert samples.shape[0] == len(pairs)
