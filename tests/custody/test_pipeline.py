from custody.pipeline import (
    Verdict,
    full_system_verdict,
    drift_only_verdict,
    coevasion_only_verdict,
    path_combined_verdict,
)


def test_full_system_verdict_no_flags():
    v = full_system_verdict(drift_flagged=False, path_break=False, coevasion_flag=None)
    assert v == Verdict(flagged=False, reasons=[])


def test_full_system_verdict_all_flags():
    v = full_system_verdict(drift_flagged=True, path_break=True, coevasion_flag="mismatch")
    assert v.flagged is True
    assert v.reasons == ["edge_drift_anomaly", "path_trajectory_break", "coevasion_mismatch"]


def test_full_system_verdict_too_perfect_flag():
    v = full_system_verdict(drift_flagged=False, path_break=False, coevasion_flag="too_perfect")
    assert v.flagged is True
    assert v.reasons == ["coevasion_too_perfect"]


def test_drift_only_verdict_ignores_coevasion():
    v = drift_only_verdict(drift_flagged=True, path_break=False)
    assert v.flagged is True
    assert v.reasons == ["edge_drift_anomaly"]


def test_coevasion_only_verdict_ignores_drift():
    v = coevasion_only_verdict(coevasion_flag="too_perfect")
    assert v.flagged is True
    assert v.reasons == ["coevasion_too_perfect"]


def test_coevasion_only_verdict_none_flag():
    v = coevasion_only_verdict(coevasion_flag=None)
    assert v == Verdict(flagged=False, reasons=[])


def test_full_system_verdict_path_elevated_flag():
    v = full_system_verdict(drift_flagged=False, path_break=False, coevasion_flag=None, path_elevated=True)
    assert v.flagged is True
    assert v.reasons == ["path_elevation_anomaly"]


def test_full_system_verdict_path_elevated_default_false():
    v = full_system_verdict(drift_flagged=False, path_break=False, coevasion_flag=None)
    assert v == Verdict(flagged=False, reasons=[])


def test_drift_only_verdict_path_elevated_flag():
    v = drift_only_verdict(drift_flagged=False, path_break=False, path_elevated=True)
    assert v.flagged is True
    assert v.reasons == ["path_elevation_anomaly"]


def test_path_combined_verdict_no_flags():
    v = path_combined_verdict(path_break=False, path_elevated=False)
    assert v == Verdict(flagged=False, reasons=[])


def test_path_combined_verdict_break_only():
    v = path_combined_verdict(path_break=True, path_elevated=False)
    assert v.flagged is True
    assert v.reasons == ["path_trajectory_break"]


def test_path_combined_verdict_elevated_only():
    v = path_combined_verdict(path_break=False, path_elevated=True)
    assert v.flagged is True
    assert v.reasons == ["path_elevation_anomaly"]


def test_path_combined_verdict_both_flags():
    v = path_combined_verdict(path_break=True, path_elevated=True)
    assert v.flagged is True
    assert v.reasons == ["path_trajectory_break", "path_elevation_anomaly"]


def test_path_combined_verdict_deficit_flag():
    v = path_combined_verdict(path_break=False, path_elevated=False, path_elevation_deficit=True)
    assert v.flagged is True
    assert v.reasons == ["path_elevation_deficit"]


def test_path_combined_verdict_deficit_default_false():
    v = path_combined_verdict(path_break=False, path_elevated=False)
    assert v == Verdict(flagged=False, reasons=[])


def test_path_combined_verdict_all_three_flags():
    v = path_combined_verdict(path_break=True, path_elevated=True, path_elevation_deficit=True)
    assert v.flagged is True
    assert v.reasons == ["path_trajectory_break", "path_elevation_anomaly", "path_elevation_deficit"]


def test_full_system_verdict_deficit_flag():
    v = full_system_verdict(
        drift_flagged=False, path_break=False, coevasion_flag=None, path_elevation_deficit=True
    )
    assert v.flagged is True
    assert v.reasons == ["path_elevation_deficit"]


def test_drift_only_verdict_deficit_flag():
    v = drift_only_verdict(drift_flagged=False, path_break=False, path_elevation_deficit=True)
    assert v.flagged is True
    assert v.reasons == ["path_elevation_deficit"]
