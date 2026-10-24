#!/bin/bash
cd "E:/DF paper" || exit 1
export PYTHONPATH=src FEATSET=v2 PAIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SP=work/degraded; mkdir -p $SP
LOCAL=0 python scripts/degraded_eval.py base $SP/native_base.json
LOCAL=1 python scripts/degraded_eval.py gbm  $SP/native_gbm.json
SRC=youtube PAIRC=work/pair_features_ytsrc.npz LOCC=work/local_features_ytsrc.npz LOCAL=0 python scripts/degraded_eval.py base $SP/yt_base.json
SRC=youtube PAIRC=work/pair_features_ytsrc.npz LOCC=work/local_features_ytsrc.npz LOCAL=1 python scripts/degraded_eval.py gbm  $SP/yt_gbm.json
echo EVALDONE
