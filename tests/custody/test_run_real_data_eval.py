from pathlib import Path

import pytest

from scripts.run_real_data_eval import build_real_data_eval_matrix

VISION_ROOT = Path(__file__).parent.parent.parent / "data" / "vision"


@pytest.mark.skipif(not VISION_ROOT.exists(), reason="VISION dataset subset not downloaded")
def test_build_real_data_eval_matrix_shape():
    result = build_real_data_eval_matrix(str(VISION_ROOT), n_attack_instances=20, seed=0)

    assert set(result.keys()) == {"whatsapp", "youtube"}
    for transform_result in result.values():
        assert set(transform_result.keys()) == {
            "snapshot_baseline",
            "drift_only",
            "coevasion_only",
            "full_system",
        }
        for method_result in transform_result.values():
            assert set(method_result.keys()) == {"naive", "heuristic", "adaptive"}
            for tier_result in method_result.values():
                assert 0.0 <= tier_result["auc"] <= 1.0
                assert 0.0 <= tier_result["f1"] <= 1.0
