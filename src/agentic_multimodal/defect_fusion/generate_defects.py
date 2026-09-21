"""批量生成由视觉专家 RAG 引导的问卷对齐 Qwen cache。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (parent for parent in CURRENT_FILE.parents if (parent / "rag_store.py").is_file()),
    CURRENT_FILE.parents[5],
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_store import ExpertQAStore

from agentic_multimodal.defect_detection import (
    DEFAULT_QWEN_MODEL,
    analyze_image_defects,
    build_defect_prompt,
    load_defect_specs,
)
from agentic_multimodal.defect_fusion.model import SURVEY_SCHEMA_VERSION
from agentic_multimodal.evidence import DEFAULT_WEIGHTS, extract_hybrid_feature, load_model


PROMPT_VERSION = "survey_visual_rag_prompt_v2"


@dataclass(frozen=True)
class ImageRow:
    """这个数据类保存一行训练标签及其逐文件 cache 目标。"""

    group_id: str
    filename: str
    image_path: Path
    output_path: Path


class RequestRateLimiter:
    """这个类按全局最小间隔调度 Qwen 请求，避免并发 worker 触发每分钟限流。"""

    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        """这个函数在需要时等待，并原子预留下一次请求启动时间。"""

        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            if delay:
                time.sleep(delay)
            self._next_allowed = time.monotonic() + self.min_interval_seconds


def _file_hash(path: str | Path) -> str:
    """这个函数计算输入文件的 SHA-256，供 cache 版本校验。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    """这个函数为稳定排序的 JSON 元数据生成 SHA-256。"""

    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_image_path(row: dict[str, str], labels_csv: Path, image_root: Path | None) -> Path | None:
    """这个函数优先使用 portable relative_path，并兼容历史路径和 augmented_classes。"""

    filename = str(row.get("filename") or row.get("image") or row.get("path") or "").strip()
    relative_path = str(row.get("relative_path") or "").strip()
    raw_path = str(row.get("combined_path") or row.get("class_path") or "").strip()
    candidates: list[Path] = []
    if image_root is not None and relative_path:
        candidates.append(image_root / Path(relative_path.replace("\\", "/")))
    if raw_path:
        candidates.append(Path(raw_path))
    if image_root is not None and filename:
        candidates.append(image_root / filename)
        if filename.lower().startswith("aug_"):
            candidates.append(image_root / "augmented_classes" / filename)
    if filename:
        candidates.extend((labels_csv.parent / filename, Path(filename)))
    for candidate in candidates:
        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = resolved.resolve()
        if resolved.is_file():
            return resolved
    return None


def _read_rows(
    labels_csv: Path,
    image_root: Path | None,
    output_dir: Path,
    reuse_original_id: bool,
) -> list[ImageRow]:
    """这个函数读取 portable 标签并为每行建立 original_id group 和 cache 路径。"""

    rows: list[ImageRow] = []
    with labels_csv.open("r", newline="", encoding="utf-8-sig") as file_handle:
        for index, row in enumerate(csv.DictReader(file_handle)):
            image_path = _resolve_image_path(row, labels_csv, image_root)
            if image_path is None:
                raise FileNotFoundError(f"Cannot resolve image for CSV row {index + 2}: {row.get('filename')}")
            filename = str(row.get("filename") or image_path.name).strip() or image_path.name
            original_id = str(row.get("original_id") or "").strip() if reuse_original_id else ""
            group_id = original_id or (image_path.stem if reuse_original_id else f"{index}:{image_path.stem}")
            rows.append(
                ImageRow(
                    group_id=group_id,
                    filename=filename,
                    image_path=image_path,
                    output_path=output_dir / f"{Path(filename).stem}.json",
                )
            )
    return rows


def _group_rows(rows: list[ImageRow]) -> dict[str, list[ImageRow]]:
    """这个函数把增强图和其原图按 original_id 归为一次 Qwen 调用。"""

    groups: dict[str, list[ImageRow]] = {}
    for row in rows:
        groups.setdefault(row.group_id, []).append(row)
    return groups


def _pick_representative(rows: list[ImageRow]) -> ImageRow:
    """这个函数优先选择非增强原图作为专家检索和 Qwen 输入。"""

    for row in rows:
        normalized_path = str(row.image_path).replace("\\", "/").lower()
        if "/augmented_classes/" not in normalized_path and not row.filename.lower().startswith("aug_"):
            return row
    return rows[0]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """这个函数通过临时文件原子写入单图 cache，防止中断留下半文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cache_matches(path: Path, row: ImageRow, metadata: dict[str, Any]) -> bool:
    """这个函数只复用 hash、schema、original_id 和增强复用标记全部一致的 cache。"""

    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return False
    required_matches = {
        "schema_version": metadata["schema_version"],
        "prompt_version": metadata["prompt_version"],
        "prompt_hash": metadata["prompt_hash"],
        "survey_config_hash": metadata["survey_config_hash"],
        "expert_qa_hash": metadata["expert_qa_hash"],
        "retrieval_index_hash": metadata["retrieval_index_hash"],
        "source_original_id": row.group_id,
    }
    if any(payload.get(key) != value for key, value in required_matches.items()):
        return False
    if not isinstance(payload.get("findings"), list) or not isinstance(payload.get("expert_evidence"), dict):
        return False
    expected_reused = row.filename.lower().startswith("aug_")
    return isinstance(payload.get("defect_reused_from_source"), bool) and (
        not expected_reused or payload["defect_reused_from_source"]
    )


def _retrieve_expert_evidence(
    row: ImageRow,
    store: ExpertQAStore,
    feature_extractor: Callable[[Path], Any],
    feature_lock: threading.Lock,
    *,
    top_k: int,
    min_similarity: float,
    exclude_current_image: bool,
) -> dict[str, Any]:
    """这个函数用冻结 CNN feature 检索专家案例并保留完整索引 provenance。"""

    with feature_lock:
        query_vector = feature_extractor(row.image_path)
    hits = store.retrieve_by_vector(
        query_vector,
        top_k=top_k,
        min_similarity=min_similarity,
        exclude_image_name=row.filename if exclude_current_image else None,
    )
    hit_payloads = [hit.to_dict() for hit in hits]
    return {
        "retrieved_cases": hit_payloads,
        "index_metadata": store.get_index_metadata(),
        "retrieval_status": "matched" if hit_payloads else "no_reliable_match",
        "query_image": row.filename,
        "top_k": top_k,
        "min_similarity": min_similarity,
        "excluded_current_image": exclude_current_image,
    }


def _payload_for_target(
    result: dict[str, Any],
    target: ImageRow,
    source: ImageRow,
    metadata: dict[str, Any],
    expert_evidence: dict[str, Any],
    qwen_status: str,
) -> dict[str, Any]:
    """这个函数把一次原图 Qwen 结果改写成逐文件、可审计且可训练的 cache。"""

    findings = result.get("findings", [])
    reused = target.image_path.resolve() != source.image_path.resolve()
    return {
        "schema_version": metadata["schema_version"],
        "prompt_version": metadata["prompt_version"],
        "prompt_hash": metadata["prompt_hash"],
        "survey_config_hash": metadata["survey_config_hash"],
        "expert_qa_hash": metadata["expert_qa_hash"],
        "retrieval_index_hash": metadata["retrieval_index_hash"],
        "cnn_checkpoint_hash": metadata["cnn_checkpoint_hash"],
        "source_original_id": target.group_id,
        "image": target.filename,
        "defect_source_image": source.filename,
        "defect_reused_from_source": reused,
        "qwen_status": qwen_status,
        "provider": result.get("provider"),
        "model": result.get("model"),
        "requested_model": result.get("requested_model"),
        "provider_error": result.get("provider_error"),
        "findings": findings,
        "defects": findings,
        "structured_output": {"defects": findings},
        "expert_evidence": expert_evidence,
        "provenance": result.get("provenance", {}),
    }


def _process_group(
    group_id: str,
    rows: list[ImageRow],
    *,
    store: ExpertQAStore,
    feature_extractor: Callable[[Path], Any],
    feature_lock: threading.Lock,
    rate_limiter: RequestRateLimiter | None,
    metadata: dict[str, Any],
    model: str,
    qwen_base_url: str | None,
    mock_response: str | None,
    top_k: int,
    min_similarity: float,
    exclude_current_image: bool,
    overwrite: bool,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    """这个函数为一个 original_id 检索专家、调用一次 Qwen 并写出全部增强行 cache。"""

    pending = [row for row in rows if overwrite or not _cache_matches(row.output_path, row, metadata)]
    representative = _pick_representative(rows)
    if not pending:
        return {"group_id": group_id, "status": "skipped_valid", "written": 0, "source": representative.filename}

    started = time.time()
    evidence = _retrieve_expert_evidence(
        representative,
        store,
        feature_extractor,
        feature_lock,
        top_k=top_k,
        min_similarity=min_similarity,
        exclude_current_image=exclude_current_image,
    )
    specs = load_defect_specs()
    result: dict[str, Any] | None = None
    for attempt in range(retries + 1):
        if rate_limiter is not None:
            rate_limiter.wait()
        result = analyze_image_defects(
            image_path=representative.image_path,
            defects=specs,
            model=model,
            qwen_base_url=qwen_base_url,
            mock_response=mock_response,
            expert_evidence=evidence,
        )
        if not result.get("provider_error"):
            break
        if attempt < retries:
            time.sleep(retry_sleep)
    if result is None or result.get("provider_error"):
        return {
            "group_id": group_id,
            "status": "qwen_failed",
            "written": 0,
            "findings": 0,
            "seconds": round(time.time() - started, 3),
            "source": representative.filename,
            "provider_error": result.get("provider_error") if result else "empty Qwen result",
        }

    qwen_status = "mock_success" if mock_response is not None else "api_success"
    for row in pending:
        _write_json(
            row.output_path,
            _payload_for_target(result, row, representative, metadata, evidence, qwen_status),
        )
    return {
        "group_id": group_id,
        "status": qwen_status,
        "written": len(pending),
        "findings": len(result.get("findings", [])),
        "seconds": round(time.time() - started, 3),
        "source": representative.filename,
        "provider_error": None,
    }


def _write_progress(path: Path, rows: list[dict[str, Any]]) -> None:
    """这个函数持续写出批处理状态，供中断恢复和失败审计。"""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group_id", "status", "written", "findings", "seconds", "source", "provider_error"]
    with path.open("w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    """这个函数构建视觉索引并并发生成可恢复的 RAG-guided Qwen cache。"""

    labels_csv = args.labels_csv.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve() if args.image_root else None
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = args.weights.expanduser().resolve()
    expert_qa = args.expert_qa.expanduser().resolve()
    survey_config = args.survey_config.expanduser().resolve()
    expert_image_root = (args.expert_image_root or image_root)
    if expert_image_root is None:
        raise ValueError("expert_image_root or image_root is required")
    expert_image_root = expert_image_root.expanduser().resolve()

    retrieval_model, retrieval_device = load_model(weights, device=args.retrieval_device)

    def feature_extractor(image_path: Path) -> Any:
        """这个闭包复用同一个冻结 CNN，为专家索引和 query 提取 feature。"""

        return extract_hybrid_feature(image_path, retrieval_model, retrieval_device)

    store = ExpertQAStore(expert_qa, cache_path=args.expert_index_cache)
    index_status = store.build_or_load_index(
        feature_extractor,
        image_root=expert_image_root,
        checkpoint_path=weights,
        survey_config_path=survey_config,
        force_rebuild=args.rebuild_expert_index,
    )
    index_metadata = store.get_index_metadata()
    specs = load_defect_specs()
    metadata = {
        "schema_version": SURVEY_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": hashlib.sha256(build_defect_prompt(specs, expert_evidence=[]).encode("utf-8")).hexdigest(),
        "survey_config_hash": _file_hash(survey_config),
        "expert_qa_hash": _file_hash(expert_qa),
        "retrieval_index_hash": _json_hash(index_metadata),
        "cnn_checkpoint_hash": _file_hash(weights),
        "index_metadata": index_metadata,
    }
    mock_response = (
        args.mock_response_file.expanduser().read_text(encoding="utf-8-sig")
        if args.mock_response_file is not None
        else None
    )

    rows = _read_rows(labels_csv, image_root, output_dir, reuse_original_id=not args.no_reuse_original_id)
    groups = _group_rows(rows)
    group_items = sorted(groups.items(), key=lambda item: item[0])
    if args.start_group:
        group_items = group_items[args.start_group :]
    if args.max_groups:
        group_items = group_items[: args.max_groups]

    progress_rows: list[dict[str, Any]] = []
    completed = written = failed = 0
    started = time.time()
    feature_lock = threading.Lock()
    rate_limiter = None if mock_response is not None else RequestRateLimiter(args.min_request_interval)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_group,
                group_id,
                group_rows,
                store=store,
                feature_extractor=feature_extractor,
                feature_lock=feature_lock,
                rate_limiter=rate_limiter,
                metadata=metadata,
                model=args.model,
                qwen_base_url=args.qwen_base_url,
                mock_response=mock_response,
                top_k=args.expert_top_k,
                min_similarity=args.expert_min_similarity,
                exclude_current_image=not args.allow_current_expert_image,
                overwrite=args.overwrite,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            ): group_id
            for group_id, group_rows in group_items
        }
        for future in as_completed(futures):
            group_id = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {
                    "group_id": group_id,
                    "status": "failed",
                    "written": 0,
                    "findings": 0,
                    "seconds": 0,
                    "source": "",
                    "provider_error": str(error),
                }
            completed += 1
            written += int(row.get("written") or 0)
            failed += int(row.get("status") in {"failed", "qwen_failed"})
            progress_rows.append(row)
            if completed == 1 or completed % args.log_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    json.dumps(
                        {
                            "completed_groups": completed,
                            "total_groups": len(group_items),
                            "written_json": written,
                            "failed_groups": failed,
                            "groups_per_minute": round(completed / elapsed * 60, 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                _write_progress(output_dir / "batch_progress.csv", progress_rows)

    _write_progress(output_dir / "batch_progress.csv", progress_rows)
    summary = {
        **metadata,
        "labels_csv": str(labels_csv),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "all_groups": len(groups),
        "selected_groups": len(group_items),
        "completed_groups": completed,
        "written_json": written,
        "failed_groups": failed,
        "workers": args.workers,
        "reuse_original_id": not args.no_reuse_original_id,
        "expert_index_status": index_status,
        "mock_mode": mock_response is not None,
        "seconds": round(time.time() - started, 3),
    }
    (output_dir / "cache_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """这个函数声明 batch cache 生成所需的 portable 数据、RAG 和 Qwen 参数。"""

    parser = argparse.ArgumentParser(description="Generate survey-aligned visual-RAG Qwen cache.")
    parser.add_argument("--labels-csv", type=Path, default=Path("combined/all_images_balanced_portable.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("combined"))
    parser.add_argument("--output-dir", type=Path, default=Path("rag_cache_survey_v2"))
    parser.add_argument("--expert-qa", type=Path, default=PROJECT_ROOT / "expert_qa.csv")
    parser.add_argument("--survey-config", type=Path, default=PROJECT_ROOT / "survey_schema_v2.json")
    parser.add_argument("--expert-image-root", type=Path)
    parser.add_argument("--expert-index-cache", type=Path, default=PROJECT_ROOT / "expert_visual_index_v2.json")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--retrieval-device", default="auto")
    parser.add_argument("--expert-top-k", type=int, default=3)
    parser.add_argument("--expert-min-similarity", type=float, default=0.75)
    parser.add_argument("--allow-current-expert-image", action="store_true")
    parser.add_argument("--rebuild-expert-index", action="store_true")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-base-url")
    parser.add_argument("--mock-response-file", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=65.0)
    parser.add_argument("--min-request-interval", type=float, default=6.5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--start-group", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-reuse-original-id", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """这个函数执行 CLI cache 生成并输出不含凭据的摘要。"""

    print(json.dumps(generate(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
