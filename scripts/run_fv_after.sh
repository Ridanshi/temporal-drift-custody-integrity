#!/bin/bash
cd "E:/DF paper" || exit 1
export PYTHONPATH=src GOP_VARIANT=minkey
while tasklist //FI "PID eq 7552" 2>/dev/null | grep -q 7552; do sleep 20; done
echo "[fv] extract_local finished, starting generation" >> work/floreview_gen.log
python scripts/floreview_xfer_gen.py >> work/floreview_gen.log 2>&1
echo ALLDONE >> work/floreview_gen.log
