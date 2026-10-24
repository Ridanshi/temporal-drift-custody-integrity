# temporal-drift-custody-integrity

Reference implementation and experiments for the manuscript
**Temporal Drift-Anomaly & Co-Evasion Detection for Chain-of-Custody Integrity in Digital Evidence**.

Existing authenticity checks for digital video compare a file's declared provenance (the manifest)
with its measured intrinsic signal at a single point in time. That snapshot view cannot see
inconsistencies that only appear across several hops, and it cannot detect an attacker who forges
both channels so that they agree. This project treats chain-of-custody authenticity as a trajectory
problem over a custody graph and provides a statistical, interpretable detector for it.

## Method at a glance

- **Per edge:** a Gaussian-mixture drift model with one baseline per declared transform type
  (`custody.drift`).
- **Whole path:** a sum-based z-test, a rank-based elevated-hop-count binomial test, and a two-sided
  extension that also flags an unnaturally low elevated-hop count (`custody.propagate`).
- **Cross-channel:** a two-sided co-evasion test based on Mahalanobis distance, flagging both
  mismatch and suspiciously perfect agreement between manifest and intrinsic signal
  (`custody.coevasion`).
- **Verdict:** aggregation of the signals above with attributed reasons (`custody.pipeline`).
- **Comparators:** a snapshot-fusion baseline, Isolation Forest and One-Class SVM
  (`custody.conflict`, `custody.ml_baseline`).
- **Evaluation:** attack tiers, metrics, and McNemar, bootstrap and calibration utilities
  (`custody.attacks`, `custody.eval`, `custody.stats`, `custody.calibration`).

Every score is an ordinary statistical quantity (a log-likelihood or a distance), so each alarm can
be traced to the number that produced it.

## Headline results

Real data: 86 VISION triplets from 15 devices (native capture, WhatsApp re-upload, YouTube
re-upload). Each cell is WhatsApp-edge AUC / YouTube-edge AUC. Attack instances are synthetic,
anchored at the real genuine mean.

```
Method                      Naive        Heuristic    Adaptive
--------------------------  -----------  -----------  -----------
Snapshot-fusion baseline    0.91 / 0.91  0.86 / 0.86  0.86 / 0.85
Drift-only ablation         1.00 / 1.00  1.00 / 1.00  0.95 / 0.94
Co-evasion-only ablation    0.79 / 0.79  0.94 / 0.94  0.45 / 0.45
Full system                 1.00 / 1.00  1.00 / 1.00  0.90 / 0.92
```

With leave-one-device-out validation across the 15 devices, the full system's mean adaptive-tier
AUC (0.90 / 0.92) exceeds the baseline's (0.85 / 0.76) and never collapses on any held-out device,
while the baseline falls to near chance on three or four devices. The paired test on the mean
difference is **not** significant at 15 devices (p = 0.372 / 0.101); a power analysis puts the
required number of folds at 141 / 41.

## Limitations

- Fifteen devices is too few for the adaptive-tier significance test to clear; more devices are
  needed.
- The adaptive attacker is synthetic. No existing dataset contains an attacker who targets this
  detector.
- On real heuristic-tier content (Celeb-DF plus a standard re-encode), detection comes entirely
  from the drift component; the co-evasion-only ablation scores at chance there.
- Multi-hop experiments use chains bootstrapped from real per-edge scores, not real multi-hop
  custody chains.

The manuscript discusses each of these in detail.

## Repository layout

- `src/custody/`: the `custody` Python package.
- `tests/`: unit tests (`pytest`). Tests that need a dataset skip automatically when it is absent.
- `scripts/`: experiment and evaluation scripts, with the result JSON files and figures they
  produced.
- `data/`: not tracked. See *Data* below.

## Installation

Python 3.11 or newer is required. Video decoding uses PyAV.

```bash
git clone https://github.com/Ridanshi/temporal-drift-custody-integrity.git
cd temporal-drift-custody-integrity
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

## Data

The experiments use three public datasets. They are large and covered by their own licences, so
they are not redistributed here.

- **VISION**: genuine native, WhatsApp and YouTube triplets (15 devices, 86 triplets used).
  https://lesc.dinfo.unifi.it/VISION/
- **DFD (Google/Jigsaw)**: real tamper instances for the naive tier (30 original/manipulated
  pairs), distributed with FaceForensics++. https://github.com/ondyari/FaceForensics
- **Celeb-DF v2**: real tamper instances for the heuristic tier (40 pairs, re-encoded with libx264
  CRF 28). https://github.com/yuezunli/celeb-deepfakeforensics

Place VISION under `data/vision/`, using one folder per device (for example
`D01_Samsung_GalaxyS3Mini`); `src/custody/vision_loader.py` documents the file naming it expects.
For DFD and Celeb-DF the loaders read a `manifest.json` listing `{"original": ..., "manipulated": ...}`
path pairs. Write absolute paths that exist on your machine. `scripts/fetch_vision_devices.py` and
`scripts/download_vision_devices.ps1` help fetch individual VISION devices.

## Reproducing the experiments

Each script prints JSON to stdout. The first argument is the data root. Several scripts still carry a
default path from the original working machine, so always pass your own.

```bash
python scripts/run_real_data_eval.py data/vision > real_data.json
```

What each script produces:

- `run_calibrated_eval.py`: the calibrated synthetic evaluation.
- `run_real_data_eval.py` and `significance_analysis.py`: the real-data AUC table and the paired
  McNemar tests.
- `run_held_out_eval.py`, `bootstrap_power_analysis.py` and `robustness_tests_n15.py`:
  leave-one-device-out validation, cluster bootstrap, power analysis, and the Wilcoxon and
  permutation checks.
- `run_real_tamper_eval.py`, `run_real_heuristic_eval.py` and `bootstrap_real_tamper_ci.py`: real
  DFD and Celeb-DF validation and their confidence intervals.
- `extended_analysis.py`: learned-baseline comparison, calibration check, threshold sweep, worked
  case study and cross-transform test.
- `calibration_experiment.py`: held-out Platt and isotonic calibration.
- `false_declaration_experiment.py`: robustness to a falsely declared transform type.
- `path_propagation_experiment.py`: the multi-hop z-test against a maximal "creeping" attacker.
- `path_elevation_experiment.py` and `path_combined_experiment.py`: the rank-based path test and the
  combined detector.
- `path_adaptive_experiment.py` and `path_two_sided_experiment.py`: an attacker that avoids the
  elevation bar, and the two-sided test that closes that gap.

The `*_result.json`, `*_n15*.json` and `*_n86*.json` files in `scripts/` are the saved outputs of
these scripts, and `fig*.png` are the figures built from them.

## Citation

If you use this code, please cite the manuscript (bibliographic details will be added on
publication):

> R. Agarwal and M. Madiajagan, "Temporal Drift-Anomaly & Co-Evasion Detection for Chain-of-Custody
> Integrity in Digital Evidence," manuscript.

## Licence

No licence has been specified yet.

## Contact

Ridanshi Agarwal, [@Ridanshi](https://github.com/Ridanshi)

## Real-file study (current manuscript)

The tables above come from the first, synthetic-attack version of the study. The current manuscript
replaces the synthetic attackers with **real forged files**: eight edits applied to decoded frames of
the VISION clips, re-encoded with real x264 pipelines at three attacker levels (stock, platform-matched
including the 30-frame GOP, and a white-box search of encoder settings). Where the two disagree, trust
the manuscript. Its main findings, with device-cluster bootstrap intervals, are in the paper; in short,
comparing a file with its source carries the detection, unsupervised drift only ties a snapshot
baseline against platform-aware forgeries, and a supervised classifier is reported as an upper bound.

Code map (all under `scripts/` and `src/custody/`):

| Step | Files |
|---|---|
| Forgery and pipelines | `src/custody/forge.py`, `scripts/gen_forgeries.py`, `scripts/gen_adaptive.py`, `scripts/gen_gop_matched.py` |
| Features | `src/custody/features_v2.py`, `src/custody/local_features.py`, `scripts/extract_*.py` |
| Detectors | `src/custody/detectors_v2.py`, `scripts/eval_v2.py`, `scripts/classifier_v2.py` |
| Established-method baselines | `scripts/baselines_established.py`, `scripts/eval_baselines.py` |
| Degraded source (YouTube copy as source) | `scripts/srcshim.py`, `scripts/degraded_eval.py`, `scripts/run_degraded_chain.sh` |
| Transfer to unseen devices (FloreView) | `scripts/download_floreview.py`, `scripts/floreview_xfer_gen.py`, `scripts/floreview_xfer_eval.py`, `scripts/floreview_xfer_report.py` |
| Real-WhatsApp pilot package | `scripts/prepare_whatsapp_pilot.py` |

Data are not in the repository. Download VISION (https://lesc.dinfo.unifi.it/VISION/) into
`data/vision` and FloreView (https://lesc.dinfo.unifi.it/FloreView/, CC BY-SA 4.0) into
`data/floreview`; scripts expect those paths and write intermediate files to `work/`. The experiments
run on CPU with FFmpeg available (the scripts use the one bundled with imageio-ffmpeg).
Install with `pip install -e .[dev]` and run the tests with `PYTHONPATH=src pytest`.

Known limits: forgeries are simple edits made by us; real platform uploads exist only for the 15 VISION
devices; the 43-device transfer set uses simulated platform encodes; there is no casework data.

### Real-WhatsApp pilot (current service)

`scripts/prepare_whatsapp_pilot.py` builds the files to send; `scripts/match_received.py`, `scripts/build_pilot_map.py`,
`scripts/real_pilot_features.py`, `scripts/real_pilot_transfer.py` and `scripts/real_pilot_eval.py` match the received
copies to their clips and evaluate them. Finding: models fitted on the VISION copies flag every copy from current
WhatsApp (854x480, very low bitrate), so detectors must be recalibrated per platform version; recalibrated
leave-one-clip-out on 11 real clips, a classifier separates the copy-move edit (AUC 0.97, 95% interval 0.87 to 1.00).
