#!/bin/bash
cd "E:/DF paper" || exit 1
export PYTHONPATH=src
bash scripts/run_degraded_chain.sh >> work/degraded_chain.log 2>&1
GOP_VARIANT=minkey python scripts/floreview_xfer_gen.py >> work/floreview_gen.log 2>&1
echo ALLDONE >> work/floreview_gen.log
