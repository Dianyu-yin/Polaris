# POLARIS main-model source code

This archive contains only the executable source for the POLARIS **Full RAG +
MLP residual** main model, two pipeline entrypoints, and its Python dependency
list. It deliberately omits all project-control files, prompt-table documents,
logs, data, model weights, Qwen caches, checkpoints, predictions, and API
credentials.

## Model path

1. The hybrid ConvNeXt-Tiny + DenseNet-121 encoder produces visual features and
   an initial PCE estimate.
2. Cosine visual retrieval selects up to three expert cases. The current image
   is excluded; the code defaults to `top_k=3` and `min_similarity=0.75`.
3. Qwen-VL receives the current EL image and serialized expert evidence, then
   returns a zero-or-one structured primary-defect JSON response.
4. The residual MLP combines CNN and defect features to predict a PCE
   correction that is added to the CNN PCE.

## Files included

```text
README.md
requirements.txt
scripts/generate_qwen_cache.sh
scripts/train_full_rag_mlp.sh
src/**/*.py
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set `PYTHONPATH=src` before running a module directly. The two shell scripts do
this automatically.

## External material you must supply

This source-only archive cannot run alone. Supply authorized paths for the
following outside the archive:

- EL labels/manifests and EL images;
- expert-case images, `expert_qa.csv`, and `survey_schema_v2.json` for cache
  generation;
- ImageNet/task-trained CNN weights and a compatible Qwen cache for training;
- `SJTU_API_KEY` only when generating a new Qwen cache.

`scripts/generate_qwen_cache.sh` documents its required environment variables
and makes external API calls. `scripts/train_full_rag_mlp.sh` documents the
inputs required by the fixed Full RAG + MLP-residual training path. Do not put
credentials, images, caches, or checkpoints into this source archive.
