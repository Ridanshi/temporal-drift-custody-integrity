# -*- coding: utf-8 -*-
R76 = ("Table 9 weakens the edits to a quarter and a half of full strength (43 clips; full strength uses all 86). Every detector loses power: drift-only falls from 0.87 to 0.80 and 0.73, and at quarter strength its interval reaches 0.59. "
"A forger who limits an edit to a quarter-strength change therefore escapes much of the margin, at the price of a weaker alteration (Figure 7).")
R77 = [
"Searching encoder settings works, and keeps working. Over 215 forged files, the share flagged by the full system falls from 74% (platform-aware starting point) to 51% after 11 encoder trials, and the mean score falls from 6.7 to 2.7 normalized units; both curves are still falling at the eleventh trial (Figure 8). "
"The attacker favors faster presets and a CRF near 21 (mean 20.6), settings that need nothing beyond public encoder options.",
"Against this attacker drift-only keeps AUC 0.86 and the snapshot baseline 0.85, while the full system drops to 0.63 and co-evasion-only to 0.29, below chance because the objective drives forgeries toward the center of the band, closer than genuine files. "
"An attacker optimizing against drift alone (110 files) lowers drift-only to 0.79 and the full system to 0.72, while the baseline stays at 0.85 because it was not targeted. At 10% false positives the baseline catches 68%, drift-only 57%, and the full system 45%. "
"Attack success is partial: the encoder knobs we allow leave the edit-consistency evidence mostly intact. Larger search budgets and content-level perturbations were not tried.",
]
R78 = ("A forger may instead declare the wrong platform (Table 10). Real YouTube files declared as WhatsApp are detected with AUC 0.86 by drift-only and 0.88 by the full system, flagged 42% and 85% of the time against 9% and 19% for honest declarations; the snapshot baseline is near chance (0.55). "
"Real WhatsApp files declared as YouTube are separated perfectly by drift-only and the full system (AUC 1.00). The co-evasion test contributes in one direction (0.83 for YouTube declared as WhatsApp, 0.41 the other way), because it sees the declared fields in which the platforms differ.")
R79 = [
"Table 11 gives flag rates on four-hop chains. On 172 genuine chains the path z-test flags 4.1% and the elevated-hop test 2.3% (one- and two-sided alike), while any-hop edge drift flags 27%, because four edges are each tested at 10%; this union inflation is a real cost of per-edge alarms on long chains. "
"With one tampered hop, edge drift flags 82% of chains, the z-test 56%, and the elevation test only 4.7%. Table 12 and Figure 9 compare them at matched false-positive rates by AUC: per-edge drift (maximum over hops) 0.86, path z-test 0.84 to 0.85, end-to-end drift between the native file and the last hop 0.66 to 0.72, and elevated-hop count 0.57 to 0.60. "
"The z-test is close to per-edge drift with a much lower flag rate on genuine chains.",
"For the creeping attacker (strength 0.34 at three hops) AUCs are lower: per-edge drift 0.76 to 0.77, z-test 0.68 at two hops rising to 0.75 at four, and elevated-hop count 0.57 rising to 0.74. The elevation test thus does what it was designed for only at four hops, and then it matches the z-test rather than beating it. "
"The two-sided elevation test flags no more chains than the one-sided one at any length, so we found no attacker that uses the low tail, and no gain from it (Table 11). Drift between the native file and the final file detects tampering less often than hop-by-hop scoring, which supports scoring every custody edge.",
]
