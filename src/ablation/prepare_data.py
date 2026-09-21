"""生成 POLARIS 正式消融实验使用的固定、无泄漏数据清单。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = (
    "filename",
    "original_id",
    "efficiency",
    "class",
    "class_path",
    "combined_path",
    "Label",
    "relative_path",
)
CLASS_ORDER = ("high", "low", "middle", "rejected")
EXPECTED_RAW_ROWS = 6448
EXPECTED_DUPLICATE_ROWS = 49
EXPECTED_CLEAN_ROWS = 6397
EXPECTED_CLEAN_GROUPS = 2461
EXPECTED_EXPERT_GROUPS = 19
DEFAULT_CONFLICT_GROUP = "20240327-38-80-AE"
TARGET_GROUP_COUNTS = {
    "train": {"high": 1462, "low": 48, "middle": 192, "rejected": 21},
    "val": {"high": 313, "low": 10, "middle": 41, "rejected": 5},
    "test": {"high": 313, "low": 10, "middle": 41, "rejected": 5},
}


def sha256_file(path: Path) -> str:
    """这个函数流式计算文件 SHA-256，用于锁定数据和输出 artifact。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_hash(value: Any) -> str:
    """这个函数为结构化配置生成稳定 hash，便于审计 split 算法。"""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """这个函数读取源 CSV，并保留每行的原始行号用于重复记录审计。"""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted(set(REQUIRED_FIELDS) - set(fieldnames))
        if missing:
            raise ValueError(f"CSV 缺少必需字段: {missing}")
        rows: list[dict[str, str]] = []
        for source_line, row in enumerate(reader, start=2):
            copied = {field: str(row.get(field, "")) for field in fieldnames}
            copied["__source_line"] = str(source_line)
            rows.append(copied)
    return fieldnames, rows


def deduplicate_exact_rows(
    rows: Iterable[dict[str, str]], fieldnames: list[str]
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """这个函数按全部原始字段逐字符串去重，并始终保留首次出现的记录。"""

    seen: dict[tuple[str, ...], int] = {}
    clean: list[dict[str, str]] = []
    duplicates: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(row[field] for field in fieldnames)
        if key in seen:
            duplicates.append(
                {
                    "original_id": row["original_id"],
                    "filename": row["filename"],
                    "class": row["class"],
                    "kept_source_line": seen[key],
                    "removed_source_line": int(row["__source_line"]),
                    "row_sha256": _stable_json_hash({field: row[field] for field in fieldnames}),
                }
            )
            continue
        seen[key] = int(row["__source_line"])
        clean.append(row)
    return clean, duplicates


def group_rows(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """这个函数按 original_id 聚合原图和增强图，确保 split 的独立单位正确。"""

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["original_id"]].append(row)
    return dict(groups)


def _is_augmented(row: dict[str, str]) -> bool:
    """这个函数识别增强图，防止 validation/test 使用 augmented sample。"""

    filename = row["filename"].strip().casefold()
    relative = row["relative_path"].replace("\\", "/").strip().casefold()
    return filename.startswith("aug_") or relative.startswith("augmented_classes/")


def choose_canonical_rows(
    groups: dict[str, list[dict[str, str]]]
) -> dict[str, dict[str, str]]:
    """这个函数为每个 original_id 唯一选择非增强原图，遇到歧义立即停止。"""

    canonical: dict[str, dict[str, str]] = {}
    for group_id, records in groups.items():
        originals = [row for row in records if not _is_augmented(row)]
        if len(originals) != 1:
            names = [row["filename"] for row in originals]
            raise ValueError(f"组 {group_id} 的 canonical original 数量为 {len(originals)}: {names}")
        canonical[group_id] = originals[0]
    return canonical


def validate_group_consistency(groups: dict[str, list[dict[str, str]]]) -> None:
    """这个函数验证组内类别和 PCE 标签一致，避免增强组内存在隐蔽冲突。"""

    problems: dict[str, dict[str, list[str]]] = {}
    for group_id, records in groups.items():
        classes = sorted({row["class"] for row in records})
        labels = sorted({row["Label"] for row in records})
        efficiencies = sorted({row["efficiency"] for row in records})
        if len(classes) != 1 or len(labels) != 1 or len(efficiencies) != 1:
            problems[group_id] = {
                "classes": classes,
                "labels": labels,
                "efficiencies": efficiencies,
            }
    if problems:
        raise ValueError(f"清洗后仍有组内标签冲突: {json.dumps(problems, ensure_ascii=False)}")


def resolve_expert_groups(
    expert_csv: Path,
    groups: dict[str, list[dict[str, str]]],
) -> tuple[list[str], dict[str, str]]:
    """这个函数用 expert_qa 图像名回连 original_id，形成强制训练组清单。"""

    with expert_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        expert_rows = list(csv.DictReader(stream))
    expert_images = sorted({str(row.get("image_name", "")).strip() for row in expert_rows})
    expert_images = [name for name in expert_images if name]

    filename_to_groups: dict[str, set[str]] = defaultdict(set)
    for group_id, records in groups.items():
        for row in records:
            filename_to_groups[row["filename"]].add(group_id)

    image_to_group: dict[str, str] = {}
    for image_name in expert_images:
        matches = sorted(filename_to_groups.get(image_name, set()))
        if len(matches) != 1:
            raise ValueError(f"专家图像 {image_name} 映射到 {len(matches)} 个组: {matches}")
        image_to_group[image_name] = matches[0]
    expert_groups = sorted(set(image_to_group.values()))
    if len(expert_images) != EXPECTED_EXPERT_GROUPS or len(expert_groups) != EXPECTED_EXPERT_GROUPS:
        raise ValueError(
            f"专家图像/组数量漂移: images={len(expert_images)}, groups={len(expert_groups)}"
        )
    return expert_groups, image_to_group


def _class_seed(seed: int, class_name: str) -> int:
    """这个函数为每一类别派生稳定随机种子，避免类别遍历顺序改变 split。"""

    digest = hashlib.sha256(f"polaris-split-v1:{seed}:{class_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def make_group_split(
    groups: dict[str, list[dict[str, str]]],
    expert_groups: Iterable[str],
    seed: int,
) -> dict[str, str]:
    """这个函数按类别随机划分组，并把全部专家知识库图像固定在 train。"""

    expert_set = set(expert_groups)
    assignment: dict[str, str] = {}
    for class_name in CLASS_ORDER:
        class_groups = sorted(
            group_id
            for group_id, records in groups.items()
            if records[0]["class"] == class_name
        )
        forced_train = sorted(set(class_groups) & expert_set)
        candidates = sorted(set(class_groups) - expert_set)
        random.Random(_class_seed(seed, class_name)).shuffle(candidates)

        n_val = TARGET_GROUP_COUNTS["val"][class_name]
        n_test = TARGET_GROUP_COUNTS["test"][class_name]
        val_groups = candidates[:n_val]
        test_groups = candidates[n_val : n_val + n_test]
        train_groups = forced_train + candidates[n_val + n_test :]

        expected_train = TARGET_GROUP_COUNTS["train"][class_name]
        if len(train_groups) != expected_train:
            raise ValueError(
                f"{class_name} train 组数 {len(train_groups)} != contract {expected_train}"
            )
        for split_name, identifiers in (
            ("train", train_groups),
            ("val", val_groups),
            ("test", test_groups),
        ):
            for group_id in identifiers:
                if group_id in assignment:
                    raise ValueError(f"组 {group_id} 被重复分配")
                assignment[group_id] = split_name

    if set(assignment) != set(groups):
        missing = sorted(set(groups) - set(assignment))
        extra = sorted(set(assignment) - set(groups))
        raise ValueError(f"split 未覆盖全部组: missing={missing}, extra={extra}")
    if any(assignment[group_id] != "train" for group_id in expert_set):
        raise ValueError("至少一个 expert group 未被固定到 train")
    return assignment


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    """这个函数以 UTF-8 和固定换行写出 portable CSV artifact。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    """这个函数用稳定格式写出 JSON audit artifact。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_image_paths(rows: Iterable[dict[str, str]], image_root: Path) -> dict[str, Any]:
    """这个函数验证 relative_path 在部署 image root 下全部存在且非空。"""

    unique_paths = {
        (image_root / row["relative_path"].replace("\\", "/")).resolve()
        for row in rows
    }
    missing = sorted(str(path) for path in unique_paths if not path.is_file())
    empty = sorted(str(path) for path in unique_paths if path.is_file() and path.stat().st_size == 0)
    return {
        "image_root": str(image_root.resolve()),
        "unique_paths": len(unique_paths),
        "missing_count": len(missing),
        "empty_count": len(empty),
        "missing_examples": missing[:20],
        "empty_examples": empty[:20],
    }


def build_manifests(
    source_csv: Path,
    expert_csv: Path,
    output_dir: Path,
    image_root: Path,
    seed: int = 42,
    conflict_group: str = DEFAULT_CONFLICT_GROUP,
) -> dict[str, Any]:
    """这个函数执行清洗、分组划分、canonical 选择和全套泄漏检查。"""

    source_csv = source_csv.expanduser().resolve()
    expert_csv = expert_csv.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, raw_rows = read_rows(source_csv)
    exact_deduped, duplicate_events = deduplicate_exact_rows(raw_rows, fieldnames)
    conflict_rows = [row for row in exact_deduped if row["original_id"] == conflict_group]
    clean_rows = [row for row in exact_deduped if row["original_id"] != conflict_group]
    groups = group_rows(clean_rows)

    if len(raw_rows) != EXPECTED_RAW_ROWS:
        raise ValueError(f"raw rows {len(raw_rows)} != contract {EXPECTED_RAW_ROWS}")
    if len(duplicate_events) != EXPECTED_DUPLICATE_ROWS:
        raise ValueError(
            f"exact duplicate rows {len(duplicate_events)} != contract {EXPECTED_DUPLICATE_ROWS}"
        )
    if len(conflict_rows) != 2 or sorted({row["Label"] for row in conflict_rows}) != ["15.83", "16.19"]:
        raise ValueError(f"冲突组内容漂移: {conflict_rows}")
    if len(clean_rows) != EXPECTED_CLEAN_ROWS or len(groups) != EXPECTED_CLEAN_GROUPS:
        raise ValueError(
            f"clean size drift: rows={len(clean_rows)}, groups={len(groups)}"
        )

    validate_group_consistency(groups)
    canonical = choose_canonical_rows(groups)
    expert_groups, expert_image_to_group = resolve_expert_groups(expert_csv, groups)
    assignment = make_group_split(groups, expert_groups, seed)
    expert_set = set(expert_groups)

    class_group_counts = Counter(records[0]["class"] for records in groups.values())
    expected_totals = {
        class_name: sum(TARGET_GROUP_COUNTS[split][class_name] for split in ("train", "val", "test"))
        for class_name in CLASS_ORDER
    }
    if dict(class_group_counts) != expected_totals:
        raise ValueError(
            f"组级类别分布漂移: actual={dict(class_group_counts)}, expected={expected_totals}"
        )

    extended_fields = fieldnames + ["split", "is_canonical", "is_expert_group"]
    extended_rows: list[dict[str, Any]] = []
    for row in clean_rows:
        group_id = row["original_id"]
        output_row = {field: row[field] for field in fieldnames}
        output_row.update(
            {
                "split": assignment[group_id],
                "is_canonical": int(row is canonical[group_id]),
                "is_expert_group": int(group_id in expert_set),
            }
        )
        extended_rows.append(output_row)

    train_rows = [row for row in extended_rows if row["split"] == "train"]
    canonical_by_group = {
        row["original_id"]: row for row in extended_rows if int(row["is_canonical"]) == 1
    }
    val_rows = [canonical_by_group[group_id] for group_id in sorted(groups) if assignment[group_id] == "val"]
    test_rows = [canonical_by_group[group_id] for group_id in sorted(groups) if assignment[group_id] == "test"]

    cleaned_path = output_dir / "cleaned_all_rows.csv"
    train_path = output_dir / "train_rows.csv"
    val_path = output_dir / "val_rows.csv"
    test_path = output_dir / "test_rows.csv"
    group_path = output_dir / "group_split.csv"
    _write_csv(cleaned_path, extended_fields, extended_rows)
    _write_csv(train_path, extended_fields, train_rows)
    _write_csv(val_path, extended_fields, val_rows)
    _write_csv(test_path, extended_fields, test_rows)

    group_fields = (
        "original_id",
        "class",
        "Label",
        "split",
        "n_rows",
        "canonical_filename",
        "canonical_relative_path",
        "is_expert_group",
    )
    group_manifest_rows = []
    for group_id in sorted(groups):
        row = canonical[group_id]
        group_manifest_rows.append(
            {
                "original_id": group_id,
                "class": row["class"],
                "Label": row["Label"],
                "split": assignment[group_id],
                "n_rows": len(groups[group_id]),
                "canonical_filename": row["filename"],
                "canonical_relative_path": row["relative_path"],
                "is_expert_group": int(group_id in expert_set),
            }
        )
    _write_csv(group_path, list(group_fields), group_manifest_rows)
    (output_dir / "expert_group_ids.txt").write_text(
        "\n".join(expert_groups) + "\n", encoding="utf-8"
    )

    split_sets = {
        split: {group_id for group_id, value in assignment.items() if value == split}
        for split in ("train", "val", "test")
    }
    actual_counts = {
        split: dict(
            Counter(groups[group_id][0]["class"] for group_id in split_sets[split])
        )
        for split in split_sets
    }
    for split in ("train", "val", "test"):
        if actual_counts[split] != TARGET_GROUP_COUNTS[split]:
            raise ValueError(
                f"{split} class counts drift: {actual_counts[split]} != {TARGET_GROUP_COUNTS[split]}"
            )

    image_validation = _validate_image_paths(clean_rows, image_root)
    if image_validation["missing_count"] or image_validation["empty_count"]:
        raise FileNotFoundError(f"图像路径验证失败: {image_validation}")

    leakage_report = {
        "status": "PASS",
        "unit": "original_id",
        "intersections": {
            "train_val": sorted(split_sets["train"] & split_sets["val"]),
            "train_test": sorted(split_sets["train"] & split_sets["test"]),
            "val_test": sorted(split_sets["val"] & split_sets["test"]),
        },
        "expert_groups_not_in_train": sorted(expert_set - split_sets["train"]),
        "val_noncanonical_rows": sum(int(row["is_canonical"]) != 1 for row in val_rows),
        "test_noncanonical_rows": sum(int(row["is_canonical"]) != 1 for row in test_rows),
        "group_counts": {split: len(group_ids) for split, group_ids in split_sets.items()},
        "class_group_counts": actual_counts,
        "image_paths": image_validation,
    }
    if any(leakage_report["intersections"].values()):
        raise ValueError(f"发现 split 泄漏: {leakage_report['intersections']}")
    _write_json(output_dir / "data_leakage_report.json", leakage_report)

    cleaning_audit = {
        "source_csv": str(source_csv),
        "source_sha256": sha256_file(source_csv),
        "exact_duplicate_definition": "all 8 source CSV fields are string-identical; keep first row",
        "duplicate_rows_removed": duplicate_events,
        "conflict_group_excluded": conflict_group,
        "conflict_rows": [
            {
                "source_line": int(row["__source_line"]),
                "filename": row["filename"],
                "efficiency": row["efficiency"],
                "Label": row["Label"],
                "class": row["class"],
            }
            for row in conflict_rows
        ],
        "row_counts": {
            "raw": len(raw_rows),
            "after_exact_dedup": len(exact_deduped),
            "after_conflict_exclusion": len(clean_rows),
        },
        "group_count_after_cleaning": len(groups),
    }
    _write_json(output_dir / "cleaning_audit.json", cleaning_audit)

    artifact_paths = [
        cleaned_path,
        train_path,
        val_path,
        test_path,
        group_path,
        output_dir / "expert_group_ids.txt",
        output_dir / "data_leakage_report.json",
        output_dir / "cleaning_audit.json",
    ]
    manifest = {
        "status": "PASS",
        "contract_version": "polaris-ablation-split-v1",
        "split_seed": seed,
        "split_algorithm": "per-class SHA256-derived Python random shuffle; expert groups removed before 15/15 allocation",
        "conflict_group_excluded": conflict_group,
        "source": {
            "csv": str(source_csv),
            "sha256": sha256_file(source_csv),
            "raw_rows": len(raw_rows),
        },
        "clean": {
            "rows": len(clean_rows),
            "groups": len(groups),
            "exact_duplicate_rows_removed": len(duplicate_events),
            "conflict_rows_removed": len(conflict_rows),
        },
        "expert": {
            "expert_csv": str(expert_csv),
            "expert_csv_sha256": sha256_file(expert_csv),
            "groups": expert_groups,
            "image_to_group": expert_image_to_group,
        },
        "splits": {
            split: {
                "groups": len(split_sets[split]),
                "rows": len(train_rows) if split == "train" else len(val_rows) if split == "val" else len(test_rows),
                "class_group_counts": actual_counts[split],
                "canonical_only": split != "train",
            }
            for split in ("train", "val", "test")
        },
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifact_paths
        },
    }
    manifest["assignment_sha256"] = _stable_json_hash(
        {group_id: assignment[group_id] for group_id in sorted(assignment)}
    )
    _write_json(output_dir / "data_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """这个函数定义正式清洗和 split 生成器的命令行接口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--expert-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conflict-group", default=DEFAULT_CONFLICT_GROUP)
    return parser.parse_args(argv)


def main() -> None:
    """这个函数运行数据准备，并仅打印不含隐私信息的最终摘要。"""

    args = parse_args()
    manifest = build_manifests(
        source_csv=args.source_csv,
        expert_csv=args.expert_csv,
        output_dir=args.output_dir,
        image_root=args.image_root,
        seed=args.seed,
        conflict_group=args.conflict_group,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

