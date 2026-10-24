# -*- coding: utf-8 -*-
H714 = "7.14 Local features and a supervised classifier"
R714 = [
"Two changes address the weaknesses above. First, local features: edits cover a few seconds and one region of the frame, but our features average over the whole clip. We therefore compare source and copy in 30-frame windows and 6 x 8 image cells, measuring error, sharpness change, residual-noise correlation and motion change, and keep how far the worst window and cell stand out from the rest of the same clip (eight features, Table 2). "
"They lift drift-only from 0.87 to 0.89 against platform-aware forgeries and from 0.86 to 0.89 against the expert forger, and blur from 0.67 to 0.72. The gain is real but modest.",
"Second, a supervised gradient-boosting classifier (depth 3, 150 trees) over all 25 features, trained on the genuine edges and the naive and platform-aware forgeries of the other devices (Table 16). "
"With local features it reaches AUC 0.97 against platform-aware forgeries and 0.96 against the white-box encoder search, well above every unsupervised detector (best 0.90), and catches 96% and 93% of those forgeries at 10% false alarms. "
"At 1% false alarms it catches only 22% and 12%, so a casework-grade operating point remains out of reach. It also generalizes across edit types: trained on four edits and tested on the fifth, with device-disjoint folds, AUC stays between 0.86 and 0.98 against platform-aware and between 0.89 and 0.97 against adaptive forgeries (Table 17), lowest for blur.",
]
R714_ATK = "ATK_PLACEHOLDER"
R714_ATK = ("The forgeries above were optimized against the earlier detectors, not against this classifier. We therefore ran the white-box search against it (110 forgeries, 11 trials, fold-specific classifiers). "
"Its AUC falls from 0.97 to 0.91 (interval 0.82 to 0.97) and its detection rate at 10% false alarms from 96% to 74%; at 1% false alarms it catches almost nothing. It stays at least level with the best unsupervised detector on the same files (drift-only 0.90, snapshot baseline 0.89), so an attacker who targets it removes most, but not all, of its advantage.")
