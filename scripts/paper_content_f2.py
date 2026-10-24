# -*- coding: utf-8 -*-
T_GB = ("Table 16. AUC (95% device-cluster bootstrap interval) of local features and the supervised classifier. Classifier columns for adaptive attacks use forgeries optimized against other detectors, except the last column.",
 ["Method", "Naive", "Platform-aware", "Adaptive (vs. full)", "Adaptive (vs. drift)", "Adaptive (vs. fusion)", "Adaptive (vs. classifier)"],
 [["Drift-only", "0.94 [0.88, 0.98]", "0.87 [0.80, 0.93]", "0.86 [0.75, 0.93]", "0.79 [0.69, 0.86]", "0.85 [0.74, 0.92]", "0.87 [0.79, 0.93]"],
  ["Drift-only + local features", "0.95 [0.86, 0.99]", "0.89 [0.79, 0.95]", "0.89 [0.76, 0.96]", "0.85 [0.72, 0.92]", "0.90 [0.78, 0.96]", "0.90 [0.79, 0.96]"],
  ["Learned fusion (Table 13)", "0.98 [0.93, 1.00]", "0.87 [0.79, 0.93]", "0.84 [0.72, 0.92]", "0.83 [0.72, 0.90]", "0.82 [0.73, 0.89]", "not run"],
  ["Gradient boosting", "0.98 [0.93, 1.00]", "0.95 [0.87, 0.99]", "0.95 [0.87, 0.99]", "0.93 [0.87, 0.98]", "0.94 [0.88, 0.98]", "0.90 [0.81, 0.97]"],
  ["Gradient boosting + local features", "0.99 [0.96, 1.00]", "0.97 [0.91, 1.00]", "0.96 [0.88, 0.99]", "0.95 [0.87, 0.99]", "0.95 [0.88, 0.99]", "0.91 [0.82, 0.97]"]])
T_LO = ("Table 17. Leave-one-edit-out AUC of the gradient-boosting classifier with local features (device-disjoint folds; the held-out edit is absent from training).",
 ["Held-out edit", "Platform-aware", "Adaptive (vs. full)", "Adaptive (vs. classifier)"],
 [["Copy-move", "0.96", "0.95", "0.90"], ["Blur", "0.86", "0.89", "0.78"], ["Splice", "0.97", "0.97", "0.93"], ["Recolor", "0.98", "0.97", "0.94"], ["Frame freeze", "0.91", "0.90", "0.80"]])
