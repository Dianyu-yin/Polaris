"""独立验证正式 seed42 retrieval 的四个 artifact 与固定抽样契约。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from ablation.retrieval_contract import (
    EXPECTED_EXPERT_CASES,
    SHUFFLE_SEED,
    choose_shuffled_case_ids,
)


EXPECTED_GROUPS = 2461
FEATURE_DIM = 1792
CONTRACT_VERSION = "polaris-retrieval-assignment-v1"
RETRIEVAL_FILENAMES = {
    "retrieval_manifest.json",
    "retrieval_assignments.jsonl",
    "expert_cases.json",
    "expert_visual_index_seed42.json",
}
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


class RetrievalValidationError(ValueError):
    """这个异常类表示 retrieval artifact 不满足正式实验契约。"""


def _require(condition: bool, message: str) -> None:
    """这个函数把契约失败统一转换为可读验证异常。"""

    if not condition:
        raise RetrievalValidationError(message)


def sha256_file(path: Path) -> str:
    """这个函数流式计算文件 SHA-256，用于核对 manifest provenance。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    """这个函数复现 retrieval builder 使用的稳定 JSON SHA-256。"""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    """这个函数读取严格 JSON，并将坏格式报告为 retrieval 验证失败。"""

    _require(path.is_file(), f"缺少 retrieval artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise RetrievalValidationError(f"无法读取 JSON {path}: {error}") from error


def _read_json_object(path: Path) -> dict[str, Any]:
    """这个函数要求 JSON 顶层必须是 object。"""

    payload = _read_json(path)
    _require(isinstance(payload, dict), f"JSON 顶层必须是 object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """这个函数读取 assignment JSONL，并拒绝空行、坏行和非 object 行。"""

    _require(path.is_file(), f"缺少 retrieval artifact: {path}")
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                _require(bool(line.strip()), f"retrieval JSONL 第 {line_number} 行为空")
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RetrievalValidationError(
                        f"retrieval JSONL 第 {line_number} 行非法: {error}"
                    ) from error
                _require(
                    isinstance(payload, dict),
                    f"retrieval JSONL 第 {line_number} 行不是 object",
                )
                records.append(payload)
    except OSError as error:
        raise RetrievalValidationError(f"无法读取 JSONL {path}: {error}") from error
    return records


def _normalized_relative_path(value: Any, label: str) -> str:
    """这个函数统一 portable 相对路径分隔符并拒绝绝对或越界路径。"""

    _require(isinstance(value, str) and value.strip(), f"{label} 为空")
    normalized = value.strip().replace("\\", "/")
    candidate = Path(normalized)
    _require(not candidate.is_absolute(), f"{label} 不是相对路径: {value!r}")
    _require(".." not in candidate.parts, f"{label} 含路径越界: {value!r}")
    return normalized


def _read_group_split(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    """这个函数读取固定 group split，并锁定 2461 个唯一 canonical group。"""

    _require(path.is_file(), f"缺少 group split: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        raise RetrievalValidationError(f"无法读取 group split {path}: {error}") from error
    _require(len(rows) == EXPECTED_GROUPS, f"group_split 行数 {len(rows)} != {EXPECTED_GROUPS}")
    ordered_ids: list[str] = []
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        group_id = str(row.get("original_id", "")).strip()
        _require(group_id and group_id not in by_id, f"group_split original_id 为空或重复: {group_id!r}")
        _require(row.get("split") in {"train", "val", "test"}, f"组 {group_id} split 非法")
        filename = str(row.get("canonical_filename", "")).strip()
        _require(filename and Path(filename).name == filename, f"组 {group_id} canonical filename 非法")
        _normalized_relative_path(row.get("canonical_relative_path"), f"组 {group_id} canonical path")
        ordered_ids.append(group_id)
        by_id[group_id] = row
    return ordered_ids, by_id


def _validate_expert_cases(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """这个函数验证 19 个 expert case 的唯一身份和图像名。"""

    cases = _read_json(path)
    _require(isinstance(cases, list), "expert_cases.json 顶层必须是 list")
    _require(len(cases) == EXPECTED_EXPERT_CASES, f"expert cases {len(cases)} != {EXPECTED_EXPERT_CASES}")
    by_id: dict[str, dict[str, Any]] = {}
    image_names: set[str] = set()
    for case in cases:
        _require(isinstance(case, dict), "expert case 必须是 object")
        case_id = str(case.get("case_id", "")).strip()
        image_name = str(case.get("image_name", "")).strip()
        _require(case_id and case_id not in by_id, f"expert case_id 为空或重复: {case_id!r}")
        _require(image_name and Path(image_name).name == image_name, f"expert image_name 非法: {image_name!r}")
        _require(image_name.casefold() not in image_names, f"expert image_name 重复: {image_name}")
        by_id[case_id] = case
        image_names.add(image_name.casefold())
    return cases, by_id


def _validate_true_top3(
    assignment: dict[str, Any],
    group_row: dict[str, str],
    case_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    """这个函数验证真实 Top-3 的身份、顺序、有限相似度和 self exclusion。"""

    group_id = str(assignment.get("group_id", ""))
    true_top3 = assignment.get("true_top3")
    _require(isinstance(true_top3, list) and len(true_top3) == 3, f"组 {group_id} true_top3 必须恰有3项")
    true_ids: list[str] = []
    similarities: list[float] = []
    current_image = str(group_row["canonical_filename"]).casefold()
    for rank, item in enumerate(true_top3, start=1):
        _require(isinstance(item, dict), f"组 {group_id} true_top3 rank{rank} 不是 object")
        case_id = str(item.get("case_id", "")).strip()
        _require(case_id in case_by_id, f"组 {group_id} true_top3 引用未知 case: {case_id!r}")
        _require(case_id not in true_ids, f"组 {group_id} true_top3 case_id 重复")
        expected_image = str(case_by_id[case_id].get("image_name", ""))
        _require(item.get("image_name") == expected_image, f"组 {group_id} true_top3 case/image identity 不一致")
        _require(expected_image.casefold() != current_image, f"组 {group_id} true_top3 包含 current expert/self")
        similarity = item.get("similarity")
        _require(type(similarity) in (int, float), f"组 {group_id} true_top3 similarity 非数字")
        score = float(similarity)
        _require(math.isfinite(score), f"组 {group_id} true_top3 similarity 非有限值")
        true_ids.append(case_id)
        similarities.append(score)
    _require(
        all(left >= right for left, right in zip(similarities, similarities[1:])),
        f"组 {group_id} true_top3 similarity 未按降序排列",
    )
    return true_ids


def _validate_assignment(
    assignment: dict[str, Any],
    group_row: dict[str, str],
    case_by_id: dict[str, dict[str, Any]],
    all_case_ids: list[str],
) -> None:
    """这个函数交叉验证一个 group 的 canonical、true 和 shuffled selection。"""

    group_id = str(assignment.get("group_id", "")).strip()
    _require(assignment.get("split") == group_row.get("split"), f"组 {group_id} split 不匹配")
    _require(
        assignment.get("canonical_filename") == group_row.get("canonical_filename"),
        f"组 {group_id} canonical filename 不匹配",
    )
    actual_path = _normalized_relative_path(
        assignment.get("canonical_relative_path"), f"组 {group_id} assignment canonical path"
    )
    expected_path = _normalized_relative_path(
        group_row.get("canonical_relative_path"), f"组 {group_id} group_split canonical path"
    )
    _require(actual_path == expected_path, f"组 {group_id} canonical path 不匹配")
    query_hash = assignment.get("query_feature_sha256")
    _require(isinstance(query_hash, str) and HASH_PATTERN.fullmatch(query_hash) is not None, f"组 {group_id} query feature SHA-256 非法")

    true_ids = _validate_true_top3(assignment, group_row, case_by_id)
    _require(assignment.get("shuffle_seed") == SHUFFLE_SEED, f"组 {group_id} shuffle seed 漂移")
    expected_selected, expected_pool = choose_shuffled_case_ids(
        group_id, all_case_ids, true_ids, SHUFFLE_SEED
    )
    selected = assignment.get("shuffled_top3_case_ids")
    pool = assignment.get("shuffled_candidate_pool")
    _require(selected == expected_selected, f"组 {group_id} shuffled Top-3 无法由固定合同重现")
    _require(pool == expected_pool, f"组 {group_id} shuffled candidate pool 无法由固定合同重现")
    _require(len(pool) == 16 and len(set(pool)) == 16, f"组 {group_id} shuffled pool 不是16个唯一 case")
    _require(not set(selected) & set(true_ids), f"组 {group_id} shuffled Top-3 与 true Top-3 重叠")
    expected_selection_hash = stable_json_hash(
        {
            "group_id": group_id,
            "true_top3": assignment.get("true_top3"),
            "shuffled_top3_case_ids": selected,
            "shuffle_seed": SHUFFLE_SEED,
        }
    )
    _require(
        assignment.get("selection_sha256") == expected_selection_hash,
        f"组 {group_id} selection SHA-256 不匹配",
    )


def _validate_index(
    path: Path,
    expert_cases: list[dict[str, Any]],
    checkpoint_sha256: str,
    manifest: dict[str, Any],
) -> None:
    """这个函数验证视觉索引为 19x1792 有限、L2-normalized 且 case identity 对齐。"""

    payload = _read_json_object(path)
    metadata = payload.get("metadata")
    _require(isinstance(metadata, dict), "visual index 缺少 metadata")
    _require(metadata.get("checkpoint_hash") == checkpoint_sha256, "visual index checkpoint SHA-256 不匹配")
    _require(manifest.get("index_metadata") == metadata, "manifest/index metadata 不一致")
    cases = payload.get("cases")
    _require(cases == expert_cases, "visual index cases 与 expert_cases.json identity 不一致")
    expected_names = [str(case["image_name"]) for case in expert_cases]
    _require(payload.get("image_names") == expected_names, "visual index image_names 与 expert cases 不一致")
    embeddings = payload.get("embeddings")
    _require(isinstance(embeddings, list) and len(embeddings) == EXPECTED_EXPERT_CASES, f"visual index 行数不是 {EXPECTED_EXPERT_CASES}")
    for row_index, row in enumerate(embeddings):
        _require(isinstance(row, list) and len(row) == FEATURE_DIM, f"visual index 第{row_index}行维度不是 {FEATURE_DIM}")
        squared_norm = 0.0
        for value in row:
            _require(type(value) in (int, float), f"visual index 第{row_index}行含非数字")
            number = float(value)
            _require(math.isfinite(number), f"visual index 第{row_index}行含非有限值")
            squared_norm += number * number
        _require(math.isclose(math.sqrt(squared_norm), 1.0, abs_tol=1e-5), f"visual index 第{row_index}行未做 L2 normalization")


def _validate_manifest(
    manifest: dict[str, Any],
    run_dir: Path,
    checkpoint: Path,
    assignments: list[dict[str, Any]],
) -> str:
    """这个函数验证 retrieval manifest 的固定参数、artifact hash 和 seed42 checkpoint。"""

    _require(manifest.get("status") == "PASS", "retrieval manifest status 不是 PASS")
    _require(manifest.get("contract_version") == CONTRACT_VERSION, "retrieval contract version 漂移")
    _require(manifest.get("groups") == EXPECTED_GROUPS, "retrieval manifest groups 漂移")
    _require(manifest.get("expert_cases") == EXPECTED_EXPERT_CASES, "retrieval manifest expert_cases 漂移")
    _require(manifest.get("true_top_k") == 3, "retrieval true_top_k 不是3")
    _require(manifest.get("min_similarity") == -1.0, "retrieval min_similarity 不是-1")
    _require(manifest.get("exclude_current_image") is True, "retrieval 未声明排除 current image")
    _require(manifest.get("shuffle_seed") == SHUFFLE_SEED, "retrieval manifest shuffle seed 漂移")
    artifacts = manifest.get("artifacts")
    expected_artifacts = RETRIEVAL_FILENAMES - {"retrieval_manifest.json"}
    _require(isinstance(artifacts, dict) and set(artifacts) == expected_artifacts, "retrieval manifest artifact 清单漂移")
    for filename in sorted(expected_artifacts):
        _require(artifacts.get(filename) == sha256_file(run_dir / filename), f"retrieval artifact SHA-256 不匹配: {filename}")
    _require(manifest.get("assignment_sha256") == stable_json_hash(assignments), "retrieval assignment SHA-256 不匹配")

    encoder = manifest.get("retrieval_encoder")
    _require(isinstance(encoder, dict), "retrieval manifest 缺少 retrieval_encoder")
    _require(encoder.get("seed") == 42, "retrieval encoder seed 不是42")
    _require(encoder.get("feature_dim") == FEATURE_DIM, f"retrieval encoder feature_dim 不是 {FEATURE_DIM}")
    _require(checkpoint.is_file(), f"缺少 seed42 CNN checkpoint: {checkpoint}")
    checkpoint_sha256 = sha256_file(checkpoint)
    _require(encoder.get("checkpoint_sha256") == checkpoint_sha256, "retrieval encoder checkpoint SHA-256 不匹配")
    recorded_checkpoint = encoder.get("checkpoint")
    _require(isinstance(recorded_checkpoint, str) and Path(recorded_checkpoint).expanduser().resolve() == checkpoint, "retrieval encoder checkpoint 路径不是正式 seed42 best.pt")
    return checkpoint_sha256


def validate_retrieval(project_root: Path) -> dict[str, Any]:
    """这个函数从 project root 独立验证完整 seed42 retrieval 产物。"""

    root = project_root.expanduser().resolve()
    _require(root.is_dir(), f"project root 不存在: {root}")
    run_dir = root / "artifacts" / "retrieval_seed42"
    _require(run_dir.is_dir(), f"retrieval_seed42 目录不存在: {run_dir}")
    for filename in RETRIEVAL_FILENAMES:
        _require((run_dir / filename).is_file(), f"缺少 retrieval artifact: {run_dir / filename}")

    group_order, groups = _read_group_split(root / "data" / "derived" / "group_split.csv")
    expert_cases, case_by_id = _validate_expert_cases(run_dir / "expert_cases.json")
    assignments = _read_jsonl(run_dir / "retrieval_assignments.jsonl")
    _require(len(assignments) == EXPECTED_GROUPS, f"retrieval assignments {len(assignments)} != {EXPECTED_GROUPS}")
    assignment_ids = [str(item.get("group_id", "")).strip() for item in assignments]
    expected_order = sorted(group_order)
    _require(assignment_ids == expected_order, "retrieval assignments 未按 original_id 排序或未精确覆盖 group_split")
    _require(len(set(assignment_ids)) == EXPECTED_GROUPS, "retrieval assignments 含重复 group_id")
    all_case_ids = sorted(case_by_id)
    for assignment in assignments:
        group_id = str(assignment.get("group_id", "")).strip()
        _require(group_id in groups, f"retrieval assignment 引用未知 group: {group_id!r}")
        _validate_assignment(assignment, groups[group_id], case_by_id, all_case_ids)

    manifest = _read_json_object(run_dir / "retrieval_manifest.json")
    checkpoint = (root / "checkpoints" / "cnn" / "seed_42" / "best.pt").resolve()
    checkpoint_sha256 = _validate_manifest(manifest, run_dir, checkpoint, assignments)
    _validate_index(
        run_dir / "expert_visual_index_seed42.json",
        expert_cases,
        checkpoint_sha256,
        manifest,
    )
    return {
        "status": "PASS",
        "project_root": str(root),
        "retrieval_dir": str(run_dir),
        "groups_verified": len(assignments),
        "expert_cases_verified": len(expert_cases),
        "feature_shape": [len(expert_cases), FEATURE_DIM],
        "seed42_checkpoint_sha256": checkpoint_sha256,
        "artifact_sha256": {
            filename: sha256_file(run_dir / filename)
            for filename in sorted(RETRIEVAL_FILENAMES)
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """这个函数定义 project-root retrieval validator 的命令行接口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def _write_output_json(path: Path, payload: dict[str, Any]) -> None:
    """这个函数原子写入可选验证报告，避免留下半写 JSON。"""

    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def main(argv: list[str] | None = None) -> int:
    """这个函数输出单行 PASS/FAIL JSON，并用退出码 0/1 表达结果。"""

    args = parse_args(argv)
    try:
        result = validate_retrieval(args.project_root)
        exit_code = 0
    except Exception as error:
        result = {
            "status": "FAIL",
            "project_root": str(args.project_root.expanduser().resolve()),
            "error": f"{type(error).__name__}: {error}",
        }
        exit_code = 1
    if args.output_json is not None:
        try:
            _write_output_json(args.output_json, result)
        except Exception as error:
            result = {
                **result,
                "status": "FAIL",
                "output_error": f"{type(error).__name__}: {error}",
            }
            exit_code = 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
