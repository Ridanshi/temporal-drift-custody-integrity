# -*- coding: utf-8 -*-
ABSTRACT = ("Forensic authenticity checks for digital video treat intake as a single-point decision and cannot see a chain of custody. "
"We model authenticity as drift over a custody graph with learned baselines per transfer type, and combine a per-edge drift model, multi-hop path tests, and a two-sided co-evasion test. "
"We evaluate on real forged files: eight tamper edits applied to decoded frames of 86 VISION clips from 15 devices, re-encoded with real x264 pipelines at three attacker levels "
"(stock re-encode, an encode matched to the public WhatsApp specification, and a white-box attacker that searches encoder settings against the deployed detector). Models and thresholds are fit on other devices only. "
"Features that compare each file with its source carry the detection. The co-evasion test helps only against stock re-encodes and header-only forgery does not rescue it, so unsupervised drift scoring only ties a snapshot baseline against platform-aware forgeries (AUC 0.87 against 0.88). "
"A supervised gradient-boosting classifier over source-versus-copy and windowed local features reaches AUC 0.97 against platform-aware forgeries and 0.96 against the white-box search, and 0.86 to 0.98 on edit types held out of training. "
"An attacker aimed at the classifier lowers it to 0.91, still level with the best unsupervised detector, and at 1% false alarms it catches at most 22% of forgeries. "
"On 430 real re-encode chains, per-edge drift (AUC 0.86) and a path z-test (0.85) detect a tampered hop. We report negative results with device-cluster bootstrap intervals.")
