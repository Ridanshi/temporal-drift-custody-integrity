# -*- coding: utf-8 -*-
T_AUC = ("Table 4. AUC (95% device-cluster bootstrap interval) of genuine WhatsApp edges against forged files, by attacker tier. The last column is the untampered-pipeline control.",
 ["Method", "Naive", "Platform-aware", "Adaptive (vs. full)", "Adaptive (vs. drift)", "Control"],
 [["Snapshot conflict", "0.84 [0.72, 0.90]", "0.88 [0.79, 0.93]", "0.85 [0.73, 0.92]", "0.85 [0.74, 0.91]", "0.54 [0.46, 0.61]"],
  ["Mismatch-only", "0.92 [0.84, 1.00]", "0.57 [0.47, 0.68]", "0.60 [0.52, 0.74]", "0.58 [0.52, 0.68]", "0.54 [0.44, 0.65]"],
  ["Drift-only", "0.94 [0.88, 0.98]", "0.87 [0.80, 0.93]", "0.86 [0.75, 0.93]", "0.79 [0.69, 0.86]", "0.61 [0.49, 0.71]"],
  ["Co-evasion-only", "0.85 [0.58, 1.00]", "0.40 [0.33, 0.47]", "0.29 [0.20, 0.36]", "0.39 [0.30, 0.46]", "0.41 [0.35, 0.47]"],
  ["Full system", "0.97 [0.92, 1.00]", "0.80 [0.74, 0.85]", "0.63 [0.52, 0.70]", "0.72 [0.63, 0.79]", "0.44 [0.36, 0.54]"],
  ["Isolation Forest", "0.86 [0.76, 0.93]", "0.80 [0.70, 0.88]", "0.81 [0.71, 0.90]", "0.77 [0.65, 0.86]", "0.47 [0.33, 0.59]"],
  ["One-Class SVM", "0.84 [0.73, 0.92]", "0.81 [0.72, 0.88]", "0.81 [0.70, 0.89]", "0.75 [0.65, 0.84]", "0.51 [0.41, 0.59]"]])
T_EDITAUC = ("Table 6. AUC by tamper edit against the platform-aware and adaptive (vs. full system) attackers.",
 ["Edit", "Plat.: baseline", "Plat.: drift", "Plat.: full", "Adapt.: baseline", "Adapt.: drift", "Adapt.: full"],
 [["Copy-move", "0.89", "0.88", "0.82", "0.86", "0.88", "0.67"], ["Blur", "0.66", "0.67", "0.50", "0.64", "0.74", "0.22"],
  ["Splice", "0.93", "0.91", "0.87", "0.91", "0.90", "0.76"], ["Recolor", "0.97", "0.94", "0.86", "0.95", "0.89", "0.72"],
  ["Frame freeze", "0.94", "0.97", "0.96", "0.88", "0.89", "0.79"]])
