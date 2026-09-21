#!/usr/bin/env bash
set -euo pipefail

# 此脚本按当前主模型的 Top-3 视觉 RAG prompt 生成新的 Qwen cache，会产生外部 API 请求。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

: "${POLARIS_LABELS_CSV:?Set POLARIS_LABELS_CSV to the authorized labels CSV.}"
: "${POLARIS_IMAGE_ROOT:?Set POLARIS_IMAGE_ROOT to the EL image directory.}"
: "${POLARIS_EXPERT_IMAGE_ROOT:?Set POLARIS_EXPERT_IMAGE_ROOT to the expert-case image directory.}"
: "${POLARIS_TASK_CNN_CHECKPOINT:?Set POLARIS_TASK_CNN_CHECKPOINT to hybrid_best.pt.}"
: "${POLARIS_OUTPUT_DIR:?Set POLARIS_OUTPUT_DIR to a new cache output directory.}"
: "${SJTU_API_KEY:?Set SJTU_API_KEY in the execution environment.}"

python -m agentic_multimodal.defect_fusion.generate_defects \
  --labels-csv "${POLARIS_LABELS_CSV}" \
  --image-root "${POLARIS_IMAGE_ROOT}" \
  --expert-image-root "${POLARIS_EXPERT_IMAGE_ROOT}" \
  --weights "${POLARIS_TASK_CNN_CHECKPOINT}" \
  --expert-qa "${ROOT}/src/expert_qa.csv" \
  --survey-config "${ROOT}/src/survey_schema_v2.json" \
  --output-dir "${POLARIS_OUTPUT_DIR}" \
  --expert-top-k 3 \
  --expert-min-similarity 0.75 \
  --model "${QWEN_MODEL:-qwen3vl}" \
  --qwen-base-url "${SJTU_BASE_URL:-https://models.sjtu.edu.cn/api/v1}"
