# -*- coding: utf-8 -*-
T_PAIR = ("Table 7. Paired difference in AUC (device-cluster bootstrap, 95% interval).",
 ["Comparison", "Naive", "Platform-aware", "Adaptive (vs. full)", "Adaptive (vs. drift)"],
 [["Full minus snapshot", "+0.13 [+0.08, +0.22]", "-0.08 [-0.13, 0.00]", "-0.22 [-0.29, -0.14]", "-0.13 [-0.21, -0.04]"],
  ["Drift minus snapshot", "+0.11 [+0.07, +0.18]", "0.00 [-0.04, +0.05]", "+0.01 [-0.04, +0.08]", "-0.06 [-0.11, +0.01]"],
  ["Full minus drift", "+0.03 [0.00, +0.06]", "-0.07 [-0.12, -0.02]", "-0.23 [-0.33, -0.14]", "-0.07 [-0.15, 0.00]"],
  ["Co-evasion minus mismatch-only", "-0.07 [-0.29, 0.00]", "-0.16 [-0.30, -0.07]", "-0.31 [-0.53, -0.21]", "-0.19 [-0.38, -0.08]"],
  ["Drift minus One-Class SVM", "+0.10 [+0.05, +0.16]", "+0.06 [+0.02, +0.11]", "+0.05 [0.00, +0.10]", "+0.04 [-0.02, +0.09]"]])
T_ABL = ("Table 8. Feature-group ablation (AUC): drift-only and full system, platform-aware and adaptive (vs. full system) attackers.",
 ["Feature set", "Plat.: drift", "Plat.: full", "Adapt.: drift", "Adapt.: full"],
 [["Declared + measured + edge (all)", "0.87", "0.80", "0.86", "0.63"], ["Without edge features", "0.67", "0.57", "0.64", "0.44"],
  ["Edge features only", "0.81", "0.81", "0.81", "0.73"], ["Declared only", "0.54", "0.43", "0.54", "0.36"],
  ["Measured + edge (no declared)", "0.85", "0.79", "0.85", "0.68"]])
T_STR = ("Table 9. AUC against edit strength, platform-aware attacker (quarter and half strength: 43 clips; full: 86 clips).",
 ["Method", "Quarter strength", "Half strength", "Full strength"],
 [["Snapshot conflict", "0.74", "0.83", "0.88"], ["Drift-only", "0.73", "0.80", "0.87"], ["Full system", "0.60", "0.70", "0.80"], ["One-Class SVM", "0.62", "0.72", "0.81"]])
