"""用正式 seed42 CNN 构建 19-case 视觉索引及 true/shuffled Top-3 分配。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ablation.train_fixed import _load_cnn_checkpoint, _resolve_device, sha256_file
from ablation.retrieval_contract import (
    EXPECTED_EXPERT_CASES,
    SHUFFLE_SEED,
    choose_shuffled_case_ids,
)
from agentic_multimodal.evidence import extract_hybrid_feature
from rag_store import ExpertQAStore


EXPECTED_GROUPS = 2461


def _json_hash(value: Any) -> str:
    """这个函数为 retrieval assignment 生成稳定 JSON hash。"""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_retrieval_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """这个函数提取全部 canonical 图像 feature，并写出固定 true/shuffled Top-3。"""

    device = _resolve_device(args.device)
    cnn, checkpoint = _load_cnn_checkpoint(args.cnn_checkpoint, device)
    if int(checkpoint.get("seed", -1)) != 42:
        raise ValueError("Retrieval encoder must be the formal seed42 CNN checkpoint")
    cnn.eval()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "expert_visual_index_seed42.json"
    store = ExpertQAStore(args.expert_csv, cache_path=index_path)

    def feature_extractor(image_source: Any) -> np.ndarray:
        """这个闭包用同一冻结 seed42 CNN 返回 1792D L2-normalized feature。"""

        return extract_hybrid_feature(image_source, cnn, device)

    index_status = store.build_or_load_visual_index(
        feature_extractor=feature_extractor,
        image_root=args.image_root,
        checkpoint_path=args.cnn_checkpoint,
        survey_config_path=args.survey_config,
        force_rebuild=True,
    )
    if len(store.cases) != EXPECTED_EXPERT_CASES:
        raise ValueError(f"Expert index cases {len(store.cases)} != {EXPECTED_EXPERT_CASES}")
    all_case_ids = sorted(str(case["case_id"]) for case in store.cases)

    with args.group_split.expanduser().resolve().open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        group_rows = list(csv.DictReader(stream))
    if len(group_rows) != EXPECTED_GROUPS:
        raise ValueError(f"group_split rows {len(group_rows)} != {EXPECTED_GROUPS}")

    assignments: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(group_rows, key=lambda item: item["original_id"]), start=1):
        group_id = row["original_id"]
        relative_path = row["canonical_relative_path"].replace("\\", "/")
        image_path = args.image_root.expanduser().resolve() / relative_path
        query_vector = feature_extractor(image_path)
        hits = store.retrieve_by_vector(
            query_vector,
            top_k=3,
            min_similarity=-1.0,
            exclude_image_name=row["canonical_filename"],
        )
        if len(hits) != 3:
            raise ValueError(f"Group {group_id} retrieved {len(hits)} true cases instead of 3")
        true_top3 = [
            {
                "case_id": hit.case_id,
                "image_name": hit.image_name,
                "similarity": float(hit.score),
            }
            for hit in hits
        ]
        true_ids = [item["case_id"] for item in true_top3]
        shuffled_ids, pool = choose_shuffled_case_ids(group_id, all_case_ids, true_ids)
        assignment = {
            "group_id": group_id,
            "split": row["split"],
            "canonical_filename": row["canonical_filename"],
            "canonical_relative_path": relative_path,
            "query_feature_sha256": hashlib.sha256(
                np.asarray(query_vector, dtype=np.float32).tobytes()
            ).hexdigest(),
            "true_top3": true_top3,
            "shuffled_top3_case_ids": shuffled_ids,
            "shuffled_candidate_pool": pool,
            "shuffle_seed": SHUFFLE_SEED,
        }
        assignment["selection_sha256"] = _json_hash(
            {
                "group_id": group_id,
                "true_top3": true_top3,
                "shuffled_top3_case_ids": shuffled_ids,
                "shuffle_seed": SHUFFLE_SEED,
            }
        )
        assignments.append(assignment)
        if args.log_every and index % args.log_every == 0:
            print(json.dumps({"processed": index, "total": len(group_rows)}), flush=True)

    assignment_path = output_dir / "retrieval_assignments.jsonl"
    with assignment_path.open("w", encoding="utf-8", newline="") as stream:
        for assignment in assignments:
            stream.write(json.dumps(assignment, ensure_ascii=False, sort_keys=True) + "\n")
    expert_cases_path = output_dir / "expert_cases.json"
    expert_cases_path.write_text(
        json.dumps(store.cases, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "PASS",
        "contract_version": "polaris-retrieval-assignment-v1",
        "retrieval_encoder": {
            "checkpoint": str(args.cnn_checkpoint.expanduser().resolve()),
            "checkpoint_sha256": sha256_file(args.cnn_checkpoint.expanduser().resolve()),
            "seed": 42,
            "feature_dim": 1792,
        },
        "index_status": index_status,
        "index_metadata": store.get_index_metadata(),
        "expert_cases": len(store.cases),
        "groups": len(assignments),
        "true_top_k": 3,
        "min_similarity": -1.0,
        "exclude_current_image": True,
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_rule": "exclude true Top-3, then sample 3 without replacement from remaining 16",
        "artifacts": {
            "expert_visual_index_seed42.json": sha256_file(index_path),
            "retrieval_assignments.jsonl": sha256_file(assignment_path),
            "expert_cases.json": sha256_file(expert_cases_path),
        },
        "assignment_sha256": _json_hash(assignments),
    }
    (output_dir / "retrieval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """这个函数定义 seed42 retrieval 构建作业的命令行接口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-checkpoint", type=Path, required=True)
    parser.add_argument("--group-split", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--expert-csv", type=Path, required=True)
    parser.add_argument("--survey-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args(argv)


def main() -> None:
    """这个函数运行 retrieval assignment 构建并打印最终 manifest。"""

    print(
        json.dumps(build_retrieval_manifest(parse_args()), ensure_ascii=False, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
