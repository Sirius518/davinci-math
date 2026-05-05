from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Any

import pyarrow.dataset as ds

from pipeline.core.io import write_canonical_record_shards
from pipeline.core.schema import CanonicalRecord
from pipeline.sources.base import BaseSourceAdapter, IngestSummary
from pipeline.utils.hashing import sha256_text


def _normalize_string(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _iter_source_rows(source_path: Path, batch_size: int):
    """Yield source rows via Arrow dataset scanning."""
    columns = ["question", "uid", "ground_truth", "dataset_name"]
    dataset = ds.dataset(source_path, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=batch_size, use_threads=True)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            yield row


def _safe_prefix(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return sanitized or "part"


def _process_single_source(
    *,
    source_path: str,
    output_dir: str,
    dataset_name_default: str,
    batch_size: int,
    target_shard_bytes: int,
    shard_prefix_base: str,
) -> dict[str, Any]:
    source = Path(source_path)
    local_count = 0
    local_prefix = f"{shard_prefix_base}-{_safe_prefix(source.stem)}"

    def iter_rows():
        nonlocal local_count
        for row in _iter_source_rows(source, batch_size):
            dataset_name = _normalize_string(row.get("dataset_name")) or dataset_name_default or source.stem
            record = CanonicalRecord(
                record_id=_normalize_string(row.get("uid")) or sha256_text(_normalize_string(row.get("question"))),
                question=_normalize_string(row.get("question")),
                raw_dataset_answer=_normalize_string(row.get("ground_truth")),
                confirmed_answer="",
                dataset_name=dataset_name,
                training_phase="",
                filter_tag="",
                verification={},
                distillation={},
                decontamination={},
                meta={},
            )
            local_count += 1
            yield record.to_storage_dict()

    shard_paths = write_canonical_record_shards(
        iter_rows(),
        output_dir,
        shard_prefix=local_prefix,
        target_shard_bytes=target_shard_bytes,
    )
    return {
        "source": str(source),
        "record_count": local_count,
        "output_paths": [str(path) for path in shard_paths],
    }


class SameFormatAdapter(BaseSourceAdapter):
    def load(self) -> list[CanonicalRecord]:
        raise NotImplementedError("Use write() for streaming same_format ingestion.")

    def write(self) -> IngestSummary:
        options = self.config.options or {}
        input_path = self.config.input
        if input_path.is_dir():
            file_names = options.get("source_files", [])
            if file_names:
                source_paths = [input_path / str(file_name) for file_name in file_names]
            else:
                file_glob = str(options.get("file_glob", "*.parquet"))
                source_paths = sorted(input_path.glob(file_glob))
        else:
            source_paths = [input_path]

        target_shard_size_mb = int(options.get("target_shard_size_mb", 200))
        target_shard_bytes = target_shard_size_mb * 1024 * 1024
        batch_size = int(options.get("batch_size", 5000))
        shard_prefix = str(options.get("shard_prefix", "same-format"))
        workers = int(options.get("workers", 1))
        clear_output = bool(options.get("clear_output", True))

        output_dir = self.config.output
        output_dir.mkdir(parents=True, exist_ok=True)
        if clear_output:
            for stale_path in output_dir.glob("*.parquet"):
                stale_path.unlink()
            for stale_path in output_dir.glob("*.json"):
                stale_path.unlink()

        if not source_paths:
            return IngestSummary(record_count=0, output_paths=[])

        total_records = 0
        all_paths: list[Path] = []

        print(f"[same_format] files={len(source_paths)} workers={max(workers, 1)} shard_target_mb={target_shard_size_mb}", flush=True)

        if workers <= 1:
            for index, source_path in enumerate(source_paths, start=1):
                result = _process_single_source(
                    source_path=str(source_path),
                    output_dir=str(output_dir),
                    dataset_name_default=self.config.dataset_name,
                    batch_size=batch_size,
                    target_shard_bytes=target_shard_bytes,
                    shard_prefix_base=shard_prefix,
                )
                total_records += int(result["record_count"])
                all_paths.extend(Path(path) for path in result["output_paths"])
                print(
                    f"[same_format] [{index}/{len(source_paths)}] {source_path.name} "
                    f"records={result['record_count']} shards={len(result['output_paths'])} (meta_data_skipped)",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _process_single_source,
                        source_path=str(source_path),
                        output_dir=str(output_dir),
                        dataset_name_default=self.config.dataset_name,
                        batch_size=batch_size,
                        target_shard_bytes=target_shard_bytes,
                        shard_prefix_base=shard_prefix,
                    ): source_path
                    for source_path in source_paths
                }
                done = 0
                for future in as_completed(futures):
                    done += 1
                    source_path = futures[future]
                    result = future.result()
                    total_records += int(result["record_count"])
                    all_paths.extend(Path(path) for path in result["output_paths"])
                    print(
                        f"[same_format] [{done}/{len(source_paths)}] {source_path.name} "
                        f"records={result['record_count']} shards={len(result['output_paths'])}"
                        f" (meta_data_skipped)",
                        flush=True,
                    )

        all_paths = sorted(all_paths)
        return IngestSummary(record_count=total_records, output_paths=all_paths)
