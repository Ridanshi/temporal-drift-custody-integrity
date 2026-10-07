from pathlib import Path

import numpy as np
import pytest

from custody.features import extract_intrinsic_features

FIXTURE = Path(__file__).parent / "fixtures" / "sample.mp4"


@pytest.mark.skipif(not FIXTURE.exists(), reason="run the ffmpeg fixture generation step first")
def test_extract_intrinsic_features_returns_fixed_length_vector():
    features = extract_intrinsic_features(str(FIXTURE))

    assert isinstance(features, np.ndarray)
    assert features.shape == (4,)
    assert np.all(np.isfinite(features))


@pytest.mark.skipif(not FIXTURE.exists(), reason="run the ffmpeg fixture generation step first")
def test_extract_intrinsic_features_deterministic():
    f1 = extract_intrinsic_features(str(FIXTURE))
    f2 = extract_intrinsic_features(str(FIXTURE))

    assert np.allclose(f1, f2)


def test_extract_intrinsic_features_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_intrinsic_features("does/not/exist.mp4")
