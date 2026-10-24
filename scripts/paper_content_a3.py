# -*- coding: utf-8 -*-
THREAT_INTRO = ("The attacker tampers with a native recording and wants the result accepted as the genuine WhatsApp transfer of that recording. "
"We evaluate three tiers of increasing sophistication, so that degradation, not a binary outcome, is the primary empirical signal:")
TIERS = [
 "Tier 1, naive: the attacker edits the video and re-encodes it with a stock encoder command (x264 CRF 23, medium preset, High profile, native resolution), with no attempt to match the platform.",
 "Tier 2, platform-aware: the attacker re-encodes with the public WhatsApp output specification (848x480 for 16:9 sources, Baseline profile, one keyframe per second) and a bit rate matched to published behavior, but has no knowledge of the detector.",
 "Tier 3, adaptive white-box: the attacker knows the deployed detector, including its fitted models and thresholds, and searches over real encoder settings (rate factor, preset, keyframe interval, deblocking, pre-filter noise or denoising, rate cap, reference frames) to minimize the detector score of its own forged file.",
]
THREAT_END = ("The attacker cannot alter the native record or the detector, and cannot change the declared resolution or profile without the detector seeing it. "
"The adaptive tier matters most for the co-evasion hypothesis: an attacker who makes the channels agree is invisible to any test that only checks mismatch.")
