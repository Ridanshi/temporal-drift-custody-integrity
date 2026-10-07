"""Tests the false-transform-declaration robustness question named as an open limitation
in Section 8.5: does an edge with a falsely declared transform type (e.g. a genuine
WhatsApp re-encode declared as a YouTube re-encode, or vice versa) still surface as
anomalous when scored against the wrong type's learned baseline, or does it evade
detection by construction?
"""
import json

import numpy as np

from custody.drift import DriftModel
from custody.vision_loader import extract_genuine_deltas

TRANSFORM_TYPES = ("whatsapp", "youtube")


def run_false_declaration_experiment(vision_root: str) -> dict:
    deltas_by_type = {t: extract_genuine_deltas(vision_root, t) for t in TRANSFORM_TYPES}

    drift_model = DriftModel(n_components=1)
    drift_model.fit(deltas_by_type, percentile=90.0)

    report = {}
    for true_type in TRANSFORM_TYPES:
        genuine_deltas = deltas_by_type[true_type]
        report[true_type] = {}
        for declared_type in TRANSFORM_TYPES:
            threshold = drift_model.thresholds[declared_type]
            scores = np.array([drift_model.score_edge(declared_type, d) for d in genuine_deltas])
            flagged = scores > threshold
            report[true_type][declared_type] = {
                "n": len(genuine_deltas),
                "n_flagged": int(flagged.sum()),
                "flag_rate": round(float(flagged.mean()), 3),
                "mean_score": round(float(scores.mean()), 3),
                "threshold": round(float(threshold), 3),
            }
    return report


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\DF paper\data\vision"
    print(json.dumps(run_false_declaration_experiment(root), indent=2))
