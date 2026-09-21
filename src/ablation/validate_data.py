"""独立验证 POLARIS 消融实验 derived_data 的完整性和无泄漏约束。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_CLEAN_ROWS = 6397
EXPECTED_CLEAN_GROUPS = 2461
EXPECTED_EXPERT_GROUPS = 19
CONFLICT_GROUP = "20240327-38-80-AE"
SPLITS = ("train", "val", "test")
EXPECTED_GROUP_COUNTS = {"train": 1723, "val": 369, "test": 369}
EXPECTED_CLASS_COUNTS = {
    "train": {"high": 1462, "low": 48, "middle": 192, "rejected": 21},
    "val": {"high": 313, "low": 10, "middle": 41, "rejected": 5},
    "test": {"high": 313, "low": 10, "middle": 41, "rejected": 5},
}
REQUIRED_ARTIFACTS = {
    "cleaned_all_rows.csv",
    "cleaning_audit.json",
    "data_leakage_report.json",
    "expert_group_ids.txt",
    "group_split.csv",
    "test_rows.csv",
    "train_rows.csv",
    "val_rows.csv",
}


class DataValidationError(ValueError):
    """这个异常类汇总 derived_data 不满足正式实验契约的原因。"""


def sha256_file(path: Path) -> str:
    """这个函数流式计算文件 SHA-256，用于核对 manifest 锁定的 artifact。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    """这个函数把所有契约失败统一转换成可读的验证异常。"""

    if not condition:
        raise DataValidationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    """这个函数读取 JSON object，并拒绝缺失文件或非 object 顶层结构。"""

    _require(path.is_file(), f"缺少文件: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"无法读取 JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON 顶层必须是 object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    """这个函数读取 CSV artifact，并保留全部字段供交叉验证。"""

    _require(path.is_file(), f"缺少文件: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(reader.fieldnames is not None, f"CSV 缺少表头: {path}")
            return [dict(row) for row in reader]
    except OSError as exc:
        raise DataValidationError(f"无法读取 CSV {path}: {exc}") from exc


def _safe_artifact_path(derived_dir: Path, name: str) -> Path:
    """这个函数限制 manifest artifact 必须直接位于 derived_data，避免路径逃逸。"""

    _require(Path(name).name == name, f"artifact 名称包含路径: {name!r}")
    path = (derived_dir / name).resolve()
    _require(os.path.commonpath((str(derived_dir), str(path))) == str(derived_dir), f"artifact 越界: {name}")
    return path


def _validate_artifact_hashes(derived_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    """这个函数验证 manifest 中全部正式 artifact 的存在性、字节数和 SHA-256。"""

    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, dict), "data_manifest.json 缺少 artifacts object")
    _require(set(artifacts) == REQUIRED_ARTIFACTS, f"artifact 清单漂移: {sorted(artifacts)}")
    verified: dict[str, str] = {}
    for name in sorted(REQUIRED_ARTIFACTS):
        contract = artifacts.get(name)
        _require(isinstance(contract, dict), f"artifact contract 非 object: {name}")
        path = _safe_artifact_path(derived_dir, name)
        _require(path.is_file(), f"artifact 不存在: {path}")
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        _require(contract.get("bytes") == actual_bytes, f"artifact 字节数不一致: {name}")
        _require(contract.get("sha256") == actual_sha256, f"artifact SHA-256 不一致: {name}")
        verified[name] = actual_sha256
    return verified


def _row_group_ids(rows: Iterable[dict[str, str]], source: str) -> set[str]:
    """这个函数提取 row artifact 的 original_id，并拒绝空组标识。"""

    group_ids = {row.get("original_id", "").strip() for row in rows}
    _require("" not in group_ids, f"{source} 含空 original_id")
    return group_ids


def _is_augmented(row: dict[str, str]) -> bool:
    """这个函数独立识别增强图，避免只信任 is_canonical 标记。"""

    filename = row.get("filename", "").strip().casefold()
    relative = row.get("relative_path", "").replace("\\", "/").strip().casefold()
    return filename.startswith("aug_") or relative.startswith("augmented_classes/")


def _validate_group_contract(
    group_rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, set[str]]]:
    """这个函数验证 2461 个组、三份 split、组交集和精确类别计数。"""

    _require(len(group_rows) == EXPECTED_CLEAN_GROUPS, f"group_split 行数应为 {EXPECTED_CLEAN_GROUPS}")
    by_id: dict[str, dict[str, str]] = {}
    split_groups = {split: set() for split in SPLITS}
    for row in group_rows:
        group_id = row.get("original_id", "").strip()
        split = row.get("split", "").strip()
        _require(group_id and group_id not in by_id, f"group_split original_id 为空或重复: {group_id!r}")
        _require(split in SPLITS, f"组 {group_id} 的 split 非法: {split!r}")
        by_id[group_id] = row
        split_groups[split].add(group_id)

    _require(CONFLICT_GROUP not in by_id, f"冲突组仍存在: {CONFLICT_GROUP}")
    intersections = {
        "train_val": split_groups["train"] & split_groups["val"],
        "train_test": split_groups["train"] & split_groups["test"],
        "val_test": split_groups["val"] & split_groups["test"],
    }
    _require(not any(intersections.values()), f"split group 交集非空: {intersections}")

    for split in SPLITS:
        actual_group_count = len(split_groups[split])
        _require(actual_group_count == EXPECTED_GROUP_COUNTS[split], f"{split} 组数 {actual_group_count} != {EXPECTED_GROUP_COUNTS[split]}")
        actual_classes = dict(Counter(by_id[group_id].get("class", "") for group_id in split_groups[split]))
        _require(actual_classes == EXPECTED_CLASS_COUNTS[split], f"{split} 类别组数 {actual_classes} != {EXPECTED_CLASS_COUNTS[split]}")
    return by_id, split_groups


def _validate_cleaned_rows(
    cleaned_rows: list[dict[str, str]], group_by_id: dict[str, dict[str, str]]
) -> None:
    """这个函数交叉验证 6397 行与 group manifest，并确认每组恰有一个 canonical。"""

    _require(len(cleaned_rows) == EXPECTED_CLEAN_ROWS, f"cleaned rows {len(cleaned_rows)} != {EXPECTED_CLEAN_ROWS}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in cleaned_rows:
        group_id = row.get("original_id", "").strip()
        _require(group_id in group_by_id, f"cleaned row 引用未知组: {group_id!r}")
        group = group_by_id[group_id]
        _require(row.get("split") == group.get("split"), f"组 {group_id} 的 split 在两个 artifact 中不一致")
        _require(row.get("class") == group.get("class"), f"组 {group_id} 的 class 在两个 artifact 中不一致")
        _require(row.get("Label") == group.get("Label"), f"组 {group_id} 的 Label 在两个 artifact 中不一致")
        grouped[group_id].append(row)

    _require(set(grouped) == set(group_by_id), "cleaned_all_rows 未精确覆盖 group_split 的全部组")
    for group_id, rows in grouped.items():
        manifest_row = group_by_id[group_id]
        _require(len(rows) == int(manifest_row.get("n_rows", "-1")), f"组 {group_id} 的 n_rows 不一致")
        canonical = [row for row in rows if row.get("is_canonical") == "1"]
        _require(len(canonical) == 1, f"组 {group_id} 的 canonical 行数为 {len(canonical)}")
        _require(canonical[0].get("filename") == manifest_row.get("canonical_filename"), f"组 {group_id} canonical filename 不一致")
        _require(canonical[0].get("relative_path") == manifest_row.get("canonical_relative_path"), f"组 {group_id} canonical relative_path 不一致")


def _validate_split_row_files(
    derived_dir: Path,
    cleaned_rows: list[dict[str, str]],
    group_by_id: dict[str, dict[str, str]],
    split_groups: dict[str, set[str]],
) -> dict[str, int]:
    """这个函数验证 train 含全部训练行，而 val/test 只含各组唯一 canonical 原图。"""

    cleaned_by_split = {
        split: [row for row in cleaned_rows if row.get("split") == split]
        for split in SPLITS
    }
    row_counts: dict[str, int] = {}
    for split in SPLITS:
        rows = _read_csv(derived_dir / f"{split}_rows.csv")
        row_counts[split] = len(rows)
        ids = _row_group_ids(rows, f"{split}_rows.csv")
        _require(ids == split_groups[split], f"{split}_rows.csv 未精确覆盖该 split 的组")
        _require(all(row.get("split") == split for row in rows), f"{split}_rows.csv 含错误 split 标记")
        if split == "train":
            expected_keys = Counter(
                (row.get("original_id"), row.get("filename"), row.get("relative_path"))
                for row in cleaned_by_split[split]
            )
            actual_keys = Counter(
                (row.get("original_id"), row.get("filename"), row.get("relative_path"))
                for row in rows
            )
            _require(actual_keys == expected_keys, "train_rows.csv 不是全部训练行的精确副本")
        else:
            _require(len(rows) == EXPECTED_GROUP_COUNTS[split], f"{split}_rows.csv 必须每组一行")
            for row in rows:
                group_id = row["original_id"]
                group = group_by_id[group_id]
                _require(row.get("is_canonical") == "1", f"{split} 组 {group_id} 使用非 canonical 行")
                _require(not _is_augmented(row), f"{split} 组 {group_id} 使用 augmented image")
                _require(row.get("filename") == group.get("canonical_filename"), f"{split} 组 {group_id} canonical filename 不匹配")
                _require(row.get("relative_path") == group.get("canonical_relative_path"), f"{split} 组 {group_id} canonical path 不匹配")
    return row_counts


def _validate_experts(
    derived_dir: Path,
    manifest: dict[str, Any],
    group_by_id: dict[str, dict[str, str]],
    train_groups: set[str],
) -> set[str]:
    """这个函数验证 19 个 expert group 在三个 artifact 中一致且全部固定在 train。"""

    expert_section = manifest.get("expert")
    _require(isinstance(expert_section, dict), "manifest 缺少 expert object")
    manifest_groups = expert_section.get("groups")
    _require(isinstance(manifest_groups, list), "manifest expert.groups 必须是 list")
    manifest_set = {str(group_id).strip() for group_id in manifest_groups}
    text_set = {
        line.strip()
        for line in (derived_dir / "expert_group_ids.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    flagged_set = {
        group_id for group_id, row in group_by_id.items() if row.get("is_expert_group") == "1"
    }
    _require(len(manifest_set) == EXPECTED_EXPERT_GROUPS, f"expert group 数量 {len(manifest_set)} != {EXPECTED_EXPERT_GROUPS}")
    _require(manifest_set == text_set == flagged_set, "expert group 在 manifest/txt/group_split 中不一致")
    _require(manifest_set <= train_groups, f"expert group 未全部进入 train: {sorted(manifest_set - train_groups)}")
    return manifest_set


def _resolve_image_path(image_root: Path, relative_path: str) -> Path:
    """这个函数将 portable relative_path 安全解析到 image root 内部。"""

    normalized = relative_path.replace("\\", "/").strip()
    _require(normalized and not Path(normalized).is_absolute(), f"relative_path 非相对路径: {relative_path!r}")
    resolved = (image_root / normalized).resolve()
    _require(os.path.commonpath((str(image_root), str(resolved))) == str(image_root), f"relative_path 越界: {relative_path!r}")
    return resolved


def _validate_images(cleaned_rows: Iterable[dict[str, str]], image_root: Path) -> int:
    """这个函数验证全部 relative image 存在且是非空普通文件。"""

    image_root = image_root.expanduser().resolve()
    _require(image_root.is_dir(), f"image root 不存在: {image_root}")
    unique_paths = {
        _resolve_image_path(image_root, row.get("relative_path", ""))
        for row in cleaned_rows
    }
    missing = sorted(str(path) for path in unique_paths if not path.is_file())
    empty = sorted(str(path) for path in unique_paths if path.is_file() and path.stat().st_size == 0)
    _require(not missing, f"relative image 缺失 {len(missing)} 个，示例: {missing[:5]}")
    _require(not empty, f"relative image 为空 {len(empty)} 个，示例: {empty[:5]}")
    return len(unique_paths)


def _validate_embedded_reports(
    manifest: dict[str, Any], leakage_report: dict[str, Any], cleaning_audit: dict[str, Any]
) -> None:
    """这个函数验证三个 JSON 报告自身声明与重新计算的固定契约一致。"""

    _require(manifest.get("status") == "PASS", "manifest status 不是 PASS")
    _require(manifest.get("contract_version") == "polaris-ablation-split-v1", "manifest contract_version 漂移")
    _require(manifest.get("split_seed") == 42, "manifest split_seed 不是 42")
    clean = manifest.get("clean", {})
    _require(clean.get("rows") == EXPECTED_CLEAN_ROWS and clean.get("groups") == EXPECTED_CLEAN_GROUPS, "manifest clean counts 漂移")
    _require(clean.get("exact_duplicate_rows_removed") in (None, 49), "manifest exact duplicate count 漂移")
    _require(clean.get("conflict_rows_removed") in (None, 2), "manifest conflict row count 漂移")
    _require(manifest.get("conflict_group_excluded") == CONFLICT_GROUP, "manifest conflict group 漂移")
    manifest_splits = manifest.get("splits")
    if manifest_splits is not None:
        _require(isinstance(manifest_splits, dict), "manifest splits 必须是 object")
        for split in SPLITS:
            split_contract = manifest_splits.get(split, {})
            _require(split_contract.get("groups") == EXPECTED_GROUP_COUNTS[split], f"manifest {split} group count 漂移")
            _require(split_contract.get("class_group_counts") == EXPECTED_CLASS_COUNTS[split], f"manifest {split} class counts 漂移")
            _require(split_contract.get("canonical_only") == (split != "train"), f"manifest {split} canonical_only 漂移")
    _require(leakage_report.get("status") == "PASS", "leakage report status 不是 PASS")
    _require(not any(leakage_report.get("intersections", {}).values()), "leakage report 记录了非空交集")
    _require(leakage_report.get("expert_groups_not_in_train") == [], "leakage report 记录 expert 未入 train")
    _require(leakage_report.get("val_noncanonical_rows") == 0, "leakage report 记录 val 非 canonical")
    _require(leakage_report.get("test_noncanonical_rows") == 0, "leakage report 记录 test 非 canonical")
    report_group_counts = leakage_report.get("group_counts")
    if report_group_counts is not None:
        _require(report_group_counts == EXPECTED_GROUP_COUNTS, "leakage report group counts 漂移")
    report_class_counts = leakage_report.get("class_group_counts")
    if report_class_counts is not None:
        _require(report_class_counts == EXPECTED_CLASS_COUNTS, "leakage report class counts 漂移")
    _require(cleaning_audit.get("conflict_group_excluded") == CONFLICT_GROUP, "cleaning audit conflict group 漂移")
    _require(cleaning_audit.get("row_counts", {}).get("after_conflict_exclusion") == EXPECTED_CLEAN_ROWS, "cleaning audit row count 漂移")
    _require(cleaning_audit.get("group_count_after_cleaning") == EXPECTED_CLEAN_GROUPS, "cleaning audit group count 漂移")


def validate_derived_data(derived_dir: Path, image_root: Path | None = None) -> dict[str, Any]:
    """这个函数执行独立、只读的完整性和无泄漏验证，并返回可审计摘要。"""

    derived_dir = derived_dir.expanduser().resolve()
    _require(derived_dir.is_dir(), f"derived_data 目录不存在: {derived_dir}")
    manifest = _read_json(derived_dir / "data_manifest.json")
    artifact_hashes = _validate_artifact_hashes(derived_dir, manifest)
    leakage_report = _read_json(derived_dir / "data_leakage_report.json")
    cleaning_audit = _read_json(derived_dir / "cleaning_audit.json")
    _validate_embedded_reports(manifest, leakage_report, cleaning_audit)

    group_rows = _read_csv(derived_dir / "group_split.csv")
    group_by_id, split_groups = _validate_group_contract(group_rows)
    cleaned_rows = _read_csv(derived_dir / "cleaned_all_rows.csv")
    _validate_cleaned_rows(cleaned_rows, group_by_id)
    split_row_counts = _validate_split_row_files(
        derived_dir, cleaned_rows, group_by_id, split_groups
    )
    expert_groups = _validate_experts(
        derived_dir, manifest, group_by_id, split_groups["train"]
    )

    if image_root is None:
        recorded_root = leakage_report.get("image_paths", {}).get("image_root")
        _require(isinstance(recorded_root, str) and recorded_root, "未提供 --image-root 且报告中无 image_root")
        image_root = Path(recorded_root)
    image_count = _validate_images(cleaned_rows, image_root)

    return {
        "status": "PASS",
        "derived_dir": str(derived_dir),
        "image_root": str(image_root.expanduser().resolve()),
        "clean_rows": len(cleaned_rows),
        "clean_groups": len(group_by_id),
        "group_counts": {split: len(split_groups[split]) for split in SPLITS},
        "class_group_counts": EXPECTED_CLASS_COUNTS,
        "split_row_counts": split_row_counts,
        "expert_groups": len(expert_groups),
        "conflict_absent": CONFLICT_GROUP not in group_by_id,
        "unique_images_verified": image_count,
        "artifact_sha256": artifact_hashes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """这个函数定义独立数据验证器的命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="图像根目录；不提供时使用 data_leakage_report.json 记录的路径",
    )
    return parser.parse_args(argv)


def main() -> None:
    """这个函数运行验证并以 JSON 输出结果，失败时由进程返回非零状态。"""

    args = parse_args()
    result = validate_derived_data(args.derived_dir, args.image_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
