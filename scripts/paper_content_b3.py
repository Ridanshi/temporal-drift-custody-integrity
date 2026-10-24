# -*- coding: utf-8 -*-
DATA_3 = ("Platform pipelines. A forged file is produced by piping the edited frames into FFmpeg 7.1 with libx264 [46, 47]. "
"We calibrated the WhatsApp specification against the real WhatsApp files (24 clips, 7 candidate settings): x264 CRF 20.5, veryfast preset, Baseline profile, keyframe interval 30, no B-frames. "
"With it, the mean per-feature bias against the real files is below 0.4 standard deviations and the variance ratio is close to 1. A YouTube-style specification "
"(High profile, CRF 25, keyframe interval 90, two B-frames; calibration grid CRF 20 to 26) is used for later hops of chains.")
DATA_4 = ("Adaptive attacker. For each forged clip the attacker starts from the platform-aware specification, then tries six random encoder settings and four local perturbations of its best setting "
"(11 encodes in total), keeping the file with the lowest score under the detector fitted on all other devices. It runs on every second clip (43 clips, 215 forgeries), against the full system. "
"A second set of 110 forgeries is optimized against the drift model alone.")
DATA_5 = ("Control. To separate tampering from pipeline mismatch we also encode the untampered native clips with the same platform-aware specification (86 files). Detectors that flag these files are flagging the pipeline, not the edit.")
DATA_6 = ("Chains. For each clip we build two four-hop chains, WhatsApp, YouTube, WhatsApp, YouTube and the reverse. Hop 1 is the real platform file; hops 2 to 4 are real re-encodes with the calibrated platform specifications "
"(172 genuine chains). Tamper chains (43 clips, three edits) apply one edit at hop 2 (129 chains), or a weaker edit (strength 0.34) at each of hops 2, 3 and 4, a creeping attacker (129 chains). "
"Later hops are not uploads to the real services, which is a limit of this study.")
