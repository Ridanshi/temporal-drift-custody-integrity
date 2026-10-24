# -*- coding: utf-8 -*-
T_FUS = ("Table 13. AUC (95% device-cluster bootstrap interval) of the learned fusion against the maximum rule and the best single detector. The last column uses forgeries optimized against the fusion itself (154 files).",
 ["Method", "Naive", "Platform-aware", "Adaptive (vs. full)", "Adaptive (vs. drift)", "Adaptive (vs. fusion)"],
 [["Maximum rule (full system)", "0.97 [0.92, 1.00]", "0.80 [0.74, 0.85]", "0.63 [0.52, 0.70]", "0.72 [0.63, 0.79]", "0.73 [0.66, 0.79]"],
  ["Learned fusion, all scores", "0.98 [0.93, 1.00]", "0.87 [0.79, 0.93]", "0.84 [0.72, 0.92]", "0.83 [0.72, 0.90]", "0.82 [0.73, 0.89]"],
  ["Learned fusion, drift + co-evasion", "0.98 [0.93, 1.00]", "0.86 [0.78, 0.92]", "0.82 [0.70, 0.89]", "0.77 [0.67, 0.84]", "0.83 [0.72, 0.90]"],
  ["Learned fusion, trained on naive only", "0.99 [0.98, 1.00]", "0.80 [0.74, 0.86]", "0.78 [0.70, 0.87]", "0.73 [0.63, 0.82]", "0.76 [0.68, 0.86]"],
  ["Best single detector", "0.97 (full)", "0.88 (baseline)", "0.86 (drift)", "0.85 (baseline)", "0.85 (drift)"]])
T_HDR = ("Table 14. AUC against stock-encoded forgeries with the declared fields rewritten (header-only forgery).",
 ["Method", "Honest header", "Clip header copied", "Median header"],
 [["Snapshot conflict", "0.84", "0.88", "0.88"], ["Mismatch-only", "0.92", "0.67", "0.66"], ["Drift-only", "0.94", "0.90", "0.93"],
  ["Co-evasion-only", "0.85", "0.42", "0.38"], ["Full system", "0.97", "0.85", "0.87"], ["Isolation Forest", "0.86", "0.79", "0.80"], ["One-Class SVM", "0.84", "0.81", "0.81"]])
T_X = ("Table 15. AUC for three further edits against the platform-aware attacker; the last column is pooled (95% interval).",
 ["Method", "Text overlay", "Inpainting", "Frame deletion", "Pooled"],
 [["Snapshot conflict", "0.97", "0.67", "0.88", "0.84 [0.75, 0.90]"], ["Drift-only", "0.97", "0.69", "0.87", "0.84 [0.77, 0.90]"],
  ["Co-evasion-only", "0.63", "0.40", "0.41", "0.48 [0.40, 0.57]"], ["Full system", "0.96", "0.54", "0.83", "0.78 [0.71, 0.82]"],
  ["One-Class SVM", "0.95", "0.57", "0.79", "0.77 [0.68, 0.84]"]])
