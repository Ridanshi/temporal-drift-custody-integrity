from pathlib import Path

import pytest

from scripts.run_real_tamper_eval import build_real_tamper_eval_matrix

VISION_ROOT = Path(__file__).parent.parent.parent / "data" / "vision"
DFD_ROOT = Path(__file__).parent.parent.parent / "data" / "dfd"


@pytest.mark.skipif(
    not (VISION_ROOT.exists() and DFD_ROOT.exists()),
    reason="VISION and/or DFD dataset subset not downloaded",
)
def test_build_real_tamper_eval_matrix_shape():
    result = build_real_tamper_eval_matrix(str(VISION_ROOT), str(DFD_ROOT))

    assert set(result.keys()) == {"whatsapp", "youtube"}
    for transform_result in result.values():
        assert set(transform_result.keys()) == {
            "snapshot_baseline",
            "drift_only",
            "coevasion_only",
            "full_system",
        }
        for method_result in transform_result.values():
            assert 0.0 <= method_result["auc"] <= 1.0
            assert 0.0 <= method_result["f1"] <= 1.0
