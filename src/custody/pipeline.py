from dataclasses import dataclass


@dataclass
class Verdict:
    flagged: bool
    reasons: list[str]


def full_system_verdict(
    drift_flagged: bool,
    path_break: bool,
    coevasion_flag: str | None,
    path_elevated: bool = False,
    path_elevation_deficit: bool = False,
) -> Verdict:
    reasons: list[str] = []
    if drift_flagged:
        reasons.append("edge_drift_anomaly")
    if path_break:
        reasons.append("path_trajectory_break")
    if path_elevated:
        reasons.append("path_elevation_anomaly")
    if path_elevation_deficit:
        reasons.append("path_elevation_deficit")
    if coevasion_flag:
        reasons.append(f"coevasion_{coevasion_flag}")
    return Verdict(flagged=bool(reasons), reasons=reasons)


def drift_only_verdict(
    drift_flagged: bool,
    path_break: bool,
    path_elevated: bool = False,
    path_elevation_deficit: bool = False,
) -> Verdict:
    reasons: list[str] = []
    if drift_flagged:
        reasons.append("edge_drift_anomaly")
    if path_break:
        reasons.append("path_trajectory_break")
    if path_elevated:
        reasons.append("path_elevation_anomaly")
    if path_elevation_deficit:
        reasons.append("path_elevation_deficit")
    return Verdict(flagged=bool(reasons), reasons=reasons)


def path_combined_verdict(
    path_break: bool, path_elevated: bool, path_elevation_deficit: bool = False
) -> Verdict:
    """The multi-hop trajectory signals, OR-combined: the sum-based z-test (path_break,
    tuned for a single large jump concentrated on one hop), the elevated-hop-count
    binomial test (path_elevated, tuned for a per-hop-bounded "creeping" attacker), and
    its low-tail counterpart (path_elevation_deficit, tuned for an attacker who avoids
    the elevated bar on every hop specifically to evade path_elevated), evaluated
    together rather than in isolation."""
    reasons: list[str] = []
    if path_break:
        reasons.append("path_trajectory_break")
    if path_elevated:
        reasons.append("path_elevation_anomaly")
    if path_elevation_deficit:
        reasons.append("path_elevation_deficit")
    return Verdict(flagged=bool(reasons), reasons=reasons)


def coevasion_only_verdict(coevasion_flag: str | None) -> Verdict:
    reasons = [f"coevasion_{coevasion_flag}"] if coevasion_flag else []
    return Verdict(flagged=bool(reasons), reasons=reasons)
