#!/bin/bash
cd "E:/DF paper" || exit 1
export PYTHONPATH=src SRC=youtube OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PAIRC=work/pair_features_ytsrc.npz LOCC=work/local_features_ytsrc.npz
PAIR=0 TIERS=simctl,platform,naive WORKERS=1 python scripts/extract_pairs.py || { echo PAIRS FAILED; exit 1; }
PAIR=0 TIERS=simctl,platform,naive WORKERS=1 python scripts/extract_local.py || { echo LOCAL FAILED; exit 1; }
echo CHAIN DONE
