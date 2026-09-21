"""构建或验证 POLARIS 清洗后 6,397 张图像的逐文件完整性清单。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable


EXPECTED_IMAGES = 6397


def _sha256_file(path: Path) -> str:
    """这个函数流式计算单张图像 SHA-256，避免把大图一次性读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_relative_paths(cleaned_csv: Path) -> list[str]:
    """这个函数从清洗 CSV 提取唯一 portable relative_path，并拒绝路径穿越。"""

    with cleaned_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    relative_paths = sorted({str(row["relative_path"]).replace("\\", "/") for row in rows})
    if len(rows) != EXPECTED_IMAGES or len(relative_paths) != EXPECTED_IMAGES:
        raise ValueError(
            f"清洗图像数量漂移: rows={len(rows)}, unique_paths={len(relative_paths)}, expected={EXPECTED_IMAGES}"
        )
    for relative_path in relative_paths:
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"relative_path 不是安全相对路径: {relative_path}")
    if "0327-38.jpg" in relative_paths:
        raise ValueError("冲突图像 0327-38.jpg 不应出现在清洗清单中")
    return relative_paths


def _describe_image(image_root: Path, relative_path: str) -> tuple[str, int, str]:
    """这个函数返回图像相对路径、字节数和内容 hash，供两端逐文件核验。"""

    path = image_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"图像为空文件: {path}")
    return relative_path, size, _sha256_file(path)


def _aggregate_sha256(records: Iterable[tuple[str, int, str]]) -> str:
    """这个函数对排序后的逐文件记录再做整体 hash，形成数据集内容指纹。"""

    digest = hashlib.sha256()
    for relative_path, size, file_hash in records:
        digest.update(f"{relative_path}\t{size}\t{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def build_manifest(
    cleaned_csv: Path,
    image_root: Path,
    output_dir: Path,
    workers: int = 4,
) -> dict[str, object]:
    """这个函数并行读取本地图像，写出可用于流式传输和远端验证的 manifest。"""

    cleaned_csv = cleaned_csv.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    relative_paths = _read_relative_paths(cleaned_csv)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        records = list(executor.map(lambda value: _describe_image(image_root, value), relative_paths))

    manifest_path = output_dir / "image_manifest.tsv"
    paths_path = output_dir / "image_paths.txt"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("relative_path", "bytes", "sha256"))
        writer.writerows(records)
    # Windows 的 Path.write_text 会把 \n 翻译为 CRLF；tar -T 会把尾部 \r 当作文件名。
    with paths_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("\n".join(relative_paths) + "\n")
    summary = {
        "status": "PASS",
        "image_root": str(image_root),
        "cleaned_csv": str(cleaned_csv),
        "images": len(records),
        "bytes": sum(record[1] for record in records),
        "aggregate_sha256": _aggregate_sha256(records),
        "manifest_sha256": _sha256_file(manifest_path),
        "paths_sha256": _sha256_file(paths_path),
    }
    (output_dir / "image_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def verify_manifest(manifest_path: Path, image_root: Path, workers: int = 4) -> dict[str, object]:
    """这个函数在目标机器重算全部图像 hash，并与传输前清单逐项比较。"""

    manifest_path = manifest_path.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    expected = [
        (row["relative_path"], int(row["bytes"]), row["sha256"])
        for row in rows
    ]
    if len(expected) != EXPECTED_IMAGES:
        raise ValueError(f"manifest images {len(expected)} != expected {EXPECTED_IMAGES}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        actual = list(
            executor.map(lambda record: _describe_image(image_root, record[0]), expected)
        )
    mismatches = [
        {
            "relative_path": expected_record[0],
            "expected_bytes": expected_record[1],
            "actual_bytes": actual_record[1],
            "expected_sha256": expected_record[2],
            "actual_sha256": actual_record[2],
        }
        for expected_record, actual_record in zip(expected, actual)
        if expected_record != actual_record
    ]
    summary = {
        "status": "PASS" if not mismatches else "FAIL",
        "images": len(actual),
        "bytes": sum(record[1] for record in actual),
        "aggregate_sha256": _aggregate_sha256(actual),
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:20],
    }
    if mismatches:
        raise ValueError(json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """这个函数定义 build/verify 两种图像清单命令。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--cleaned-csv", type=Path, required=True)
    build.add_argument("--image-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--workers", type=int, default=4)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--image-root", type=Path, required=True)
    verify.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main() -> None:
    """这个函数执行图像清单构建或验证，并打印机器可读摘要。"""

    args = parse_args()
    if args.command == "build":
        result = build_manifest(args.cleaned_csv, args.image_root, args.output_dir, args.workers)
    else:
        result = verify_manifest(args.manifest, args.image_root, args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
