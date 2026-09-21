#!/usr/bin/env bash
# End-to-end pipeline: cache -> multiplicative flow -> predict -> WFC beamforming.
#
# The two stages live in different Python environments on this machine
# (torch in py310, jax in dbua), so the driver shells out to each.  Override
# with PY_TORCH / PY_JAX / SAMPLES / CACHE / OUT.
set -euo pipefail

PY_TORCH=${PY_TORCH:-/home/zhuangyang/miniconda3/envs/py310/bin/python}
PY_JAX=${PY_JAX:-/home/zhuangyang/miniconda3/envs/dbua/bin/python}
SAMPLES=${SAMPLES:-/data/zhuangyang/openbreast_pw_iq/dataset/samples}
CACHE=${CACHE:-cache}
OUT=${OUT:-out}
EPOCHS=${EPOCHS:-300}
SPEEDS=${SPEEDS:-1450,1500,1550}
N_SUBAP=${N_SUBAP:-4}
WORKERS=${WORKERS:-32}
N_PREDICT=${N_PREDICT:-8}      # test samples sent to WFC beamforming
SKIP_CACHE=${SKIP_CACHE:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}

cd "$(dirname "$0")/.."
echo "== 1/4 preprocess IQ -> cache (phase-preserving DAS condition + SoS target) =="
if [ "$SKIP_CACHE" != "1" ]; then
  "$PY_TORCH" scripts/prepare_cache.py --samples "$SAMPLES" --cache "$CACHE" \
      --workers "$WORKERS" --speeds "$SPEEDS" --n-subap "$N_SUBAP" \
      --groups full,angle,subap
fi

echo "== 2/4 train multiplicative flow matching =="
if [ "$SKIP_TRAIN" != "1" ]; then
  "$PY_TORCH" train.py --config configs/sos_flow.json --cache "$CACHE" \
      --out "$OUT/flow" --epochs "$EPOCHS" --workers "$WORKERS"
fi

echo "== 3/4 predict SoS maps on the test split =="
"$PY_TORCH" predict.py --ckpt "$OUT/flow/best.pth" --cache "$CACHE" \
    --split test --out "$OUT/pred_flow"

echo "== 4/4 WFC beamforming with the predicted maps =="
"$PY_JAX" wfc_integration/beamform.py --samples "$SAMPLES" \
    --cmaps "$OUT/pred_flow/cmaps" --out "$OUT/wfc" --limit "$N_PREDICT"

echo "done -> $OUT/wfc (summary.csv, panels, per-sample npz)"
