from pathlib import Path

import pytest

from scripts.extended_analysis import run_all

VISION_ROOT = Path(__file__).parent.parent.parent / "data" / "vision"


@pytest.mark.skipif(not VISION_ROOT.exists(), reason="VISION dataset subset not downloaded")
def test_run_all_shape():
    result = run_all(str(VISION_ROOT), n_attack_instances=15, seed=0)

    assert set(result.keys()) == {
        "ml_baseline_comparison",
        "calibration",
        "threshold_sensitivity",
        "case_study",
        "cross_transform_generalization",
    }
    assert set(result["ml_baseline_comparison"].keys()) == {"whatsapp", "youtube"}
    assert set(result["calibration"].keys()) == {"whatsapp", "youtube"}
    assert "matched_fit_on_youtube_auc" in result["cross_transform_generalization"]
    assert "cross_fit_whatsapp_scoring_youtube_auc" in result["cross_transform_generalization"]
    assert "verdict_reasons" in result["case_study"]
