# Derived data and saved results

Small derived files that back the tables in the manuscript. No raw video is included.

- `derived_data/whatsapp_pilot_features.json` per-file feature vectors of the real current-WhatsApp pilot (Section 7.18).
- `derived_data/real_match_zip1.csv`, `real_pilot_map2.csv`, `real_pilot_strict_clips.json`, `whatsapp_pilot_manifest.csv`: content-based matching of received WhatsApp files to FloreView clips and conditions, and the strict clip list.
- `derived_data/floreview_transfer_features.json` features of the 43-device FloreView transfer test (Section 7.17).
- `results/` saved evaluation outputs (degraded-source, transfer, real-pilot and adaptive-attack summaries).

Raw videos are not redistributed: VISION and FloreView are public datasets (FloreView under CC BY-SA 4.0), and the pilot copies are derived from FloreView clips. Forged files are regenerated with the scripts in `scripts/`.
