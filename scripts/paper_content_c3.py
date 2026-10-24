# -*- coding: utf-8 -*-
DISC = [
("8.1 What the evidence supports",
 "Three results survive every check. First, source-versus-target consistency is the signal: edge features alone give AUC 0.81, and removing them costs 0.20. Second, device-disjoint calibration gives honest false-positive rates (9% to 14% against 10% nominal), so the numbers are not inflated by fitting on the test device. "
 "Third, a path z-test on per-hop drift scores approaches per-edge drift on real chains at a much lower genuine flag rate. Two results go against the design: the co-evasion test does not help against platform-aware, adaptive or header-only forgeries, and the rank-based elevated-hop test adds little on real chains. A learned fusion of the component scores recovers the best single detector in every tier, which the maximum rule does not, and a supervised classifier with local features goes well beyond it (AUC 0.97 and 0.96), until an attacker aims at it (0.91)."),
("8.2 Why the two-sided test fails here",
 "The low-tail argument assumes that an attacker who makes declared and measured properties agree lands closer to the center than natural capture does. In our setting both channels are outputs of the same encode, so a platform-aware forger agrees with the platform automatically, and agreement is typical of genuine transfers too: 2.6% of forged files fall in the low tail, fewer than the 7% of genuine files. "
 "The genuine distribution is also broad, because it pools 15 devices. Header-only forgery, where declared fields are rewritten after encoding, does not rescue it either (Section 7.11): the measured and edge features still reveal the stock encode. The low tail might still matter with a per-device reference, which we did not test. "
 "The evaluation does show that the two-sided score should not be merged with drift by taking the larger value when the attacker controls the encoder."),
("8.3 Limitations",
 "(1) The data are 86 clips from 15 devices and one platform pair, and intervals are wide where it matters, in the adaptive tier. (2) Forgeries are five simple edits applied by us, not professional manipulations or deepfakes. "
 "(3) Only the first hop of each chain is a real platform file; later hops use calibrated encoder settings rather than uploads, and a trained classifier can tell the simulation from the real service (AUC 0.75). "
 "(4) The adaptive attacker tries 11 encoder settings (31 on a subset) against a static detector; content-level perturbations and adaptive attacks on chains were not studied, and the attack on the learned fusion covers 154 forgeries. (5) Edge-consistency features require the source file to be retained, and the learned fusion and the classifier need labeled forgeries from other devices, and the classifier attack covers 110 forgeries. "
 "(6) Thresholds target a 10% false-positive rate, too high for casework. (7) Features use the first 180 frames only. (8) An earlier extended set of six whole-clip inconsistency features did not help, whereas the windowed local features of Section 7.14 do, modestly. "
 "(9) Scores are uncalibrated rankings; calibrating them to probabilities [39] would be needed before quoting a likelihood to a court. (10) Two of 102 unit tests fail where the dataset manifests do not match local paths."),
("8.4 Implications for practice",
 "Three practical points follow. Keep the source file at every custody step, because comparing a file with its source is the strongest signal we found, and drift scores computed without it lose most of their power. "
 "Score every custody edge and test the path as well, since a path z-test flags 4% of genuine chains where any-hop edge drift flags 27%. "
 "Treat an alarm as a reason for examination, not a finding: a 10% false-positive rate and an AUC around 0.9 against informed attackers are far from casework standards, and an attacker who knows the detector can already lower the full-system detection rate with a few encoder trials."),
]
CONCL = ("We treated chain-of-custody authenticity as drift over a custody graph and evaluated it on real forged files with device-disjoint calibration. "
"Detection comes from comparing the transferred file with its source (AUC 0.87 against a platform-aware attacker, 0.86 against a white-box encoder search), not from the two-sided co-evasion test, which helps only against stock re-encodes and is not rescued by header-only forgery. "
"A supervised gradient-boosting classifier over source-versus-copy and windowed local features is the strongest detector (AUC 0.97 and 0.96) and generalizes to held-out edit types (0.86 to 0.98), but an attacker aimed at it lowers it to 0.91, level with the best unsupervised detector, and at 1% false alarms it catches at most 22% of forgeries. "
"On real re-encode chains, per-edge drift and a path z-test detect a tampered hop (AUC 0.86 and 0.84 to 0.85). Search effort still pays, since 31 encoder trials cut the share of forgeries flagged from 81% to 63%. "
"Open questions are per-device references, real uploads for chain hops, content-level attackers, a calibrated low-false-alarm operating point, and larger and more varied data.")
AVAIL = ("The source code, forgery and evaluation scripts, saved experiment outputs, and figure scripts are openly available at https://github.com/Ridanshi/temporal-drift-custody-integrity [45]. "
"The experiments use the public VISION dataset [6], which is not redistributed here because of its size and license; forged files are regenerated by the released scripts with FFmpeg and x264 [46, 47].")
REFS_NEW = [
"[46] FFmpeg developers, “FFmpeg: a complete, cross-platform solution to record, convert and stream audio and video,” version 7.1, 2024. [Online]. Available: https://ffmpeg.org/ (accessed Oct. 2026).",
"[47] VideoLAN, “x264: a free software library and application for encoding video streams into the H.264/MPEG-4 AVC format.” [Online]. Available: https://www.videolan.org/developers/x264.html (accessed Oct. 2026).",
]
