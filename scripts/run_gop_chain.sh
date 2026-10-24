#!/bin/bash
# Waits for the GOP-matched encodes, then extracts features, baseline features and detector scores, and runs the evaluation.
cd "E:/DF paper" || exit 1
export PYTHONPATH=src
LOG="${TEMP:-/tmp}/gop_chain.log"
SP="$1"
until grep -q "^done" "$TEMP/gop_enc.out" 2>/dev/null; do sleep 15; done
echo "[chain] encodes done: $(ls work/forged/platform_gop | wc -l) platform_gop, $(ls work/forged/simctl_gop | wc -l) simctl_gop" | tee -a "$LOG"
WORKERS=1 python scripts/extract_gop_matched.py >> "$LOG" 2>&1 || { echo "[chain] extract FAILED"; exit 1; }
echo "[chain] features done" | tee -a "$LOG"
BASE_TIERS=platform_gop,simctl_gop BASE_OUT=work/baselines_gop WORKERS=2 python scripts/baselines_established.py >> "$LOG" 2>&1 || { echo "[chain] baselines FAILED"; exit 1; }
echo "[chain] baseline features done: $(ls work/baselines_gop | wc -l) clips" | tee -a "$LOG"
FEATSET=v2 PAIR=1 LOCAL=0 python scripts/eval_gop_scores.py base "$SP/p4_base.pkl" >> "$LOG" 2>&1 || { echo "[chain] scores base FAILED"; exit 1; }
FEATSET=v2 PAIR=1 LOCAL=1 python scripts/eval_gop_scores.py gbm "$SP/p4_gbm.pkl" >> "$LOG" 2>&1 || { echo "[chain] scores gbm FAILED"; exit 1; }
echo "[chain] scores done" | tee -a "$LOG"
python scripts/eval_baselines.py "$SP/p3_base.pkl" "$SP/p3_gbm.pkl" "$SP/baselines_eval_final.json" "$SP/p4_base.pkl" "$SP/p4_gbm.pkl" > "$SP/baselines_eval_final.txt" 2>&1 || { echo "[chain] eval FAILED"; tail -5 "$SP/baselines_eval_final.txt"; exit 1; }
echo "[chain] ALL DONE"
