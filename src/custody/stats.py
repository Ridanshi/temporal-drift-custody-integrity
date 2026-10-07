from math import erfc, sqrt

import numpy as np


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired McNemar test (continuity-corrected) on boolean per-instance
    correctness arrays for two methods evaluated on the same instances.
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        chi2 = 0.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = erfc(sqrt(chi2 / 2))
    return {"b": b, "c": c, "chi2": float(chi2), "p_value": float(p_value)}
