# -*- coding: utf-8 -*-
T_FD = ("Table 10. False transfer-type declaration: AUC of honestly declared against falsely declared genuine files, and full-system flag rate (honest / false).",
 ["Case", "Baseline", "Drift-only", "Co-evasion", "Full system", "Full flag rate"],
 [["YouTube declared as WhatsApp", "0.55", "0.86", "0.83", "0.88", "19% / 85%"], ["WhatsApp declared as YouTube", "0.58", "1.00", "0.41", "1.00", "19% / 100%"]])
T_CH = ("Table 11. Flag rates on four-hop chains (172 genuine, 129 with one tampered hop, 129 creeping).",
 ["Chains", "Edge drift (any hop)", "Path z-test", "Elevation, one-sided", "Elevation, two-sided", "End-to-end drift"],
 [["Genuine", "27.3%", "4.1%", "2.3%", "2.3%", "8.7%"], ["One tampered hop", "82.2%", "55.8%", "4.7%", "4.7%", "29.5%"], ["Creeping, three hops", "70.5%", "40.3%", "17.1%", "17.1%", "30.2%"]])
T_CHAUC = ("Table 12. Chain-level AUC (genuine against tampered chains) by number of hops.",
 ["Statistic", "One hop: 2", "One hop: 3", "One hop: 4", "Creeping: 2", "Creeping: 3", "Creeping: 4"],
 [["Per-edge drift (max)", "0.86", "0.86", "0.86", "0.76", "0.77", "0.77"], ["Path z-test", "0.85", "0.85", "0.84", "0.68", "0.75", "0.75"],
  ["End-to-end drift", "0.68", "0.72", "0.66", "0.59", "0.68", "0.65"], ["Elevated-hop count", "0.60", "0.57", "0.58", "0.57", "0.70", "0.74"]])
CAPS = {
 "figA": "Figure 4. AUC by attacker tier and method with 95% device-cluster bootstrap intervals (Table 4). The dashed line is chance.",
 "figB": "Figure 5. ROC curves against the platform-aware (left) and adaptive (right) attacker, pooled over held-out devices.",
 "figC": "Figure 6. AUC by tamper edit for the platform-aware (left) and adaptive (right) attacker (Table 6).",
 "figE": "Figure 7. AUC against edit strength for the platform-aware attacker (Table 9).",
 "figF": "Figure 8. Adaptive attacker progress: mean full-system score of its best forgery (left) and share still flagged (right) against the number of encoder trials.",
 "figD": "Figure 9. Chain-level AUC for one tampered hop (left) and a creeping attacker (right), by chain length (Table 12).",
}
