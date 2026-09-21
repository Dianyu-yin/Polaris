#!/usr/bin/env bash
set -euo pipefail

# 此脚本按固定协议训练 Full RAG + MLP residual；它不会生成或替换 Qwen cache。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

: "${POLARIS_TRAIN_CSV:?Set POLARIS_TRAIN_CSV to the locked train manifest.}"
: "${POLARIS_VAL_CSV:?Set POLARIS_VAL_CSV to the locked validation manifest.}"
: "${POLARIS_IMAGE_ROOT:?Set POLARIS_IMAGE_ROOT to the EL image directory.}"
: "${POLARIS_TASK_CNN_CHECKPOINT:?Set POLARIS_TASK_CNN_CHECKPOINT to same-seed CNN best.pt.}"
: "${POLARIS_CACHE_ROOT:?Set POLARIS_CACHE_ROOT to the directory containing legacy_full_rag.}"
: "${POLARIS_OUTPUT_DIR:?Set POLARIS_OUTPUT_DIR to a new run directory.}"
: "${POLARIS_SEED:?Set POLARIS_SEED.}"

python -m ablation.train_fixed \
  --stage fusion \
  --condition full_rag_mlp_residual \
  --seed "${POLARIS_SEED}" \
  --train-csv "${POLARIS_TRAIN_CSV}" \
  --val-csv "${POLARIS_VAL_CSV}" \
  --image-root "${POLARIS_IMAGE_ROOT}" \
  --cnn-checkpoint "${POLARIS_TASK_CNN_CHECKPOINT}" \
  --cache-root "${POLARIS_CACHE_ROOT}" \
  --output-dir "${POLARIS_OUTPUT_DIR}" \
  --device "${POLARIS_DEVICE:-cuda}" \
  --run-scope "${POLARIS_RUN_SCOPE:-formal}" \
  --epochs 20 \
  --batch-size 4 \
  --num-workers 6 \
  --lr 1e-4 \
  --cnn-lr 1e-5 \
  --weight-decay 1e-4 \
  --classification-loss-weight 0.2 \
  --cnn-regression-loss-weight 0.5 \
  --freeze-cnn-epochs 3 \
  --dropout 0.2 \
  --amp \
  --log-every-steps 25
