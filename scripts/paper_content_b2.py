# -*- coding: utf-8 -*-
AGG_P = ("The full system takes the larger of the normalized drift score and the co-evasion band score, so it fires when either does. "
"We evaluate the two components alone (drift-only, co-evasion-only), a mismatch-only test that keeps just the high tail of the co-evasion test, and Isolation Forest and One-Class SVM on the same drift vectors. "
"A supervised gradient-boosting classifier over the same features is evaluated in Section 7.14. In single-hop experiments the path component is inactive; Section 7.9 evaluates it on multi-hop chains.")

BASE_P = ("Following Xiang and Horvath [1], the snapshot-conflict baseline is the Euclidean norm of the standardized feature difference between declared reference and measured state, with no custody history and no per-transform model. "
"It is the closest prior-art comparison for the problem as we frame it (Table 1) and appears in every result.")

PATH_EXTRA = (" We also count hops whose drift score exceeds the 70th percentile of their own edge type and apply a one-sided binomial test, with a two-sided version that splits the significance level between tails. "
"Edge types are the declared platform and whether the source is the original capture or an earlier transfer.")

DATA_1 = ("Genuine data. The VISION dataset [6] supplies 86 native recordings from 15 devices, each with the real WhatsApp and YouTube transfer of the same recording "
"(258 files, about 15 GB). Native clips are mostly 1920x1080 at about 17 Mbit/s; the real WhatsApp files are 848x480 (or the native size for 4:3 sources), Baseline profile, about 1.3 Mbit/s, one keyframe per second; "
"the real YouTube files keep the native resolution, use High or Main profile, and average 2.3 Mbit/s. All features use the first 180 frames (6 s) of each file, so that real and forged files are comparable.")
DATA_2 = ("Tamper edits. Five edits are applied to the decoded frames 50 to 140 of a native clip before re-encoding (Table 3). They are simple, reproducible edits that stand in for localized, global and temporal manipulations; "
"they are not professional forgeries, and no deepfake is involved. Three further edits (overlaid text, inpainting removal, frame deletion) are evaluated in Section 7.12.")
