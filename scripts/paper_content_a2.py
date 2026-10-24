# -*- coding: utf-8 -*-
INTRO_FRAME = ("We reframe chain-of-custody authenticity as a temporal, distributional anomaly-detection problem over a custody graph. "
"Nodes are evidence states; edges are declared transfers, each with an expected drift distribution learned from genuine examples of that transfer type. "
"We detect anomalies at three levels: per-edge drift (is this transition typical for its declared type), multi-hop path consistency (does the trajectory hold together), "
"and a two-sided co-evasion test (is the relationship between declared and measured properties natural, or broken, or suspiciously perfect). "
"The two-sided design targets attackers who defeat snapshot detectors by making the channels agree too well. Figure 1 summarizes the pipeline.")

INTRO_PLAN = ("Section 3 defines the threat model. Section 4 gives the statistical formulation, restricted to standard primitives (Gaussian mixtures, Mahalanobis distance, binomial tests) "
"because a forensic tool whose reasoning cannot be explained to a court is a liability regardless of its accuracy. "
"Earlier versions of this work evaluated on simulated feature vectors; that is easy to get wrong, because a synthetic attacker can perturb dimensions that no real encoder can touch. "
"Here every forgery is a real file. We tamper with decoded frames of VISION videos [6], re-encode them with FFmpeg and x264 [46, 47], and score them with detectors that never saw the device. "
"Sections 5 and 6 describe the forgery pipeline and protocol; Section 7 reports results, including those that go against our design hypothesis.")

CONTRIB = ("(1) A formulation of chain-of-custody authenticity as drift over a custody graph rather than a snapshot decision. "
"(2) A drift model, a two-sided co-evasion test, and rank-based multi-hop path tests, all built from standard statistical primitives so that alarms stay interpretable. "
"(3) A real-forgery evaluation protocol: tamper edits applied to real video, real x264 platform pipelines calibrated to observed WhatsApp and YouTube outputs, a white-box adaptive attacker that searches encoder settings, "
"real re-encode chains, device-disjoint calibration, and an untampered-pipeline control. "
"(4) Empirical findings with device-cluster bootstrap intervals, including negative ones: source-versus-target consistency features carry detection; the co-evasion test helps against stock re-encodes but not against platform-aware, adaptive or header-only forgeries; a learned fusion removes the maximum rule's penalty, and a supervised classifier over source-versus-copy and local features is the strongest detector, though an attacker aimed at it removes most of its lead. "
"(5) Public code (see Data and code availability).")
