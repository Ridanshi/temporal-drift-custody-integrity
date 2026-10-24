# -*- coding: utf-8 -*-
H710 = "7.10 Learned fusion of the component scores"
R710 = [
"Taking the larger of two scores is what pulled the full system down. A learned fusion removes that penalty. We train a logistic regression on the six component scores (drift, co-evasion, mismatch-only, snapshot conflict, Isolation Forest, One-Class SVM) using the genuine edges and the naive and platform-aware forgeries of the other devices, with nested leave-device-out scoring so that training scores are out of sample. "
"It reaches AUC 0.98, 0.87, 0.84 and 0.83 on the four tiers (Table 13), within 0.02 of the best single detector in every tier and ahead of it against stock re-encodes. At 10% false positives it catches 66% of adaptive forgeries, against 45% for the maximum rule and 68% for the snapshot baseline.",
"It does not beat the best single detector against informed attackers, so the gain is robustness rather than power: it is never meaningfully worse than the best component, whichever attacker appears. Two caveats apply. The fusion is supervised and assumes labeled forgeries from other devices; trained on stock re-encodes alone it generalizes less well (0.80 against platform-aware and 0.78 against adaptive forgeries). "
"And the adaptive forgeries above were optimized against the maximum rule, not against the fusion.",
]
H711 = "7.11 Header-only forgery"
R711 = ("The co-evasion hypothesis is strongest when an attacker can edit declared fields after encoding. We simulate that on the stock-encoded forgeries by overwriting the declared bit rate, pixel count, frame rate and profile with the genuine WhatsApp values of the same clip, or with the median WhatsApp values (Table 14). "
"The co-evasion test falls to 0.42 and 0.38 and the mismatch-only test to 0.67 and 0.66, while drift-only stays at 0.90 and 0.93: the measured and edge features still reveal the stock encode, so forging the header does not make the two-sided test useful. It does cost the attacker something, since the full system drops from 0.97 to 0.85 and 0.87.")
H712 = "7.12 Three further edits"
R712 = ("Table 15 adds three edits to the five above: overlaid text, inpainting removal of a region, and deletion of 30 frames. Overlaid text is easiest (drift-only 0.97), frame deletion is detected at 0.87, and inpainting is hardest (0.69), like blurring because it removes detail while leaving the measured statistics close to natural. "
"Pooled over the three, drift-only (0.84) ties the snapshot baseline (0.84) and the full system and co-evasion test stay below, the same ordering as for the original edits.")

H713 = "7.13 Larger attacker budgets and an attacker aimed at the fusion"
R713 = [
"We extended the adaptive search from 11 to 31 encoder trials on 43 of the forgeries. The share flagged by the full system falls from 81% at the platform-aware starting point to 70% after 11 trials and 63% after 31, and the mean score from 3.4 to 2.6 (Figure 10). "
"Returns diminish but do not stop within this budget.",
"We also optimized 154 forgeries against the learned fusion itself. Its AUC falls only from 0.84 to 0.82 (interval 0.73 to 0.89), with 68% of forgeries caught at 10% false positives, in line with the snapshot baseline (0.84) and drift-only (0.85) on the same files, while the maximum rule reaches 0.73 and the co-evasion test 0.35 (Table 13, last column). "
"The learned fusion is therefore robust to an attacker who targets it, within the search budget we tried.",
]
CAP_G = "Figure 10. Share of forgeries flagged by the full system against the number of encoder trials (43 forgeries); the dashed line marks the 11 trials of the main experiment."
