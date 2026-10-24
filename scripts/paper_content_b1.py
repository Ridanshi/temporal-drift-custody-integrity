# -*- coding: utf-8 -*-
FEAT_P = ("Features. Each file is described by 12 per-file features in two groups (Table 2), computed over its first 180 frames. "
"Declared features come from container and stream headers: log bit rate, log pixel count, frame rate, and codec profile. "
"Measured features come from decoded pixels and packet sizes: log noise-residual standard deviation, log residual kurtosis, spatial noise inconsistency, log 8x8 blockiness, DCT high-frequency ratio, "
"log temporal noise, log measured bits per pixel, and log I-to-P packet-size ratio. "
"Each edge additionally has five edge-consistency features that compare the transferred file with its source at the same frame positions: mean PSNR, its temporal standard deviation, "
"the concentration of error in one image cell, the largest temporal jump in error, and the fraction of frames with unusually low PSNR. "
"Eight windowed local features (Section 7.14) add how far the worst 30-frame window and image cell stand out in error, sharpness, noise correlation and motion. These need the earlier custody state, which a chain of custody retains by definition. Earlier versions of this work used four toy statistics, one of which (frame count modulo 100) carried 97.5% of the drift variance; "
"here the categorical profile field is excluded from the drift vector and no other feature carries more than 15% of the delta variance.")

DRIFT_P = ("Thresholds are the 90th percentile of cross-validated genuine scores, computed leave-one-device-out inside the training devices (Section 6). "
"The drift vector is the standardized difference between transferred and source features (declared and measured features plus the five edge-consistency features), "
"modeled by a one-component full-covariance mixture with covariance regularization 0.01.")

COEV_P = "We learn the distribution of the 12 per-file feature vectors of genuine files of the declared transfer type and score a new file by shrinkage (Ledoit-Wolf) Mahalanobis distance [8] to its center:"
COEV_P2 = ("tau_high and tau_low are the 95th and 5th percentiles of cross-validated genuine distances (Figure 3). "
"Flagging the low tail is what distinguishes this from the manifest-versus-intrinsic tests we found: an attacker who forges both channels to agree should land closer to the center than capture noise allows. "
"For evaluation we use a continuous signed band score, the larger of (d - tau_high) and (tau_low - d), scaled by the genuine spread; it is positive exactly when the two-sided test fires and keeps ranking information for AUC.")
