from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from pipeline.core.schema import (
    CANONICAL_RECORD_SCHEMA,
    DEDUP_JUDGEMENT_SCHEMA,
    DEDUP_PAIR_SCHEMA,
    PROCESSOR_RESULT_SCHEMA,
    ROLLOUT_RESULT_SCHEMA,
    CanonicalRecord,
    DedupJudgement,
    DedupPair,
    ProcessorResult,
    RolloutResult,
)
from pipeline.utils.jsonx import dumps as json_dumps


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


@dataclass(slots=True)
class ProjectLayout:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def manifests_dir(self) -> Path:
        return self.artifacts_dir / "manifests"

    def ensure(self) -> None:
        for path in [
            self.root / "data" / "raw",
            self.root / "data" / "intermediate",
            self.root / "data" / "processed",
            self.root / "artifacts" / "cache",
            self.root / "artifacts" / "judgements",
            self.root / "artifacts" / "rollouts",
            self.root / "artifacts" / "manifests",
            self.root / "reports",
        ]:
            path.mkdir(parents=True, exist_ok=True)


def discover_layout(start: str | Path) -> ProjectLayout:
    return ProjectLayout(root=Path(start).resolve())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class StageSummary:
    name: str
    input_path: str
    output_path: str
    input_records: int
    output_records: int
    dropped_records: int
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunManifest:
    run_id: str
    created_at: str
    pipeline_name: str
    config_path: str
    input_paths: list[str]
    stage_summaries: list[StageSummary] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        pipeline_name: str,
        config_path: str,
        input_paths: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> "RunManifest":
        return cls(
            run_id=run_id,
            created_at=_utc_now(),
            pipeline_name=pipeline_name,
            config_path=config_path,
            input_paths=input_paths,
            metadata=metadata or {},
        )

    def add_stage_summary(self, summary: StageSummary) -> None:
        self.stage_summaries.append(summary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "pipeline_name": self.pipeline_name,
            "config_path": self.config_path,
            "input_paths": self.input_paths,
            "stage_summaries": [asdict(item) for item in self.stage_summaries],
            "metadata": self.metadata,
        }

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json_dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _table_from_rows(rows: list[dict[str, object]], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema)


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist([], schema=schema)


def _records_to_table(records: list[CanonicalRecord]) -> pa.Table:
    arrays = {
        "record_id": pa.array([record.record_id for record in records], type=pa.string()),
        "question": pa.array([record.question for record in records], type=pa.string()),
        "raw_dataset_answer": pa.array([record.raw_dataset_answer for record in records], type=pa.string()),
        "confirmed_answer": pa.array([record.confirmed_answer for record in records], type=pa.string()),
        "dataset_name": pa.array([record.dataset_name for record in records], type=pa.string()),
        "training_phase": pa.array([record.training_phase for record in records], type=pa.string()),
        "filter_tag": pa.array([record.filter_tag for record in records], type=pa.string()),
        "verification": pa.array(
            [json_dumps(dict(record.verification), ensure_ascii=False, sort_keys=True) for record in records],
            type=pa.large_string(),
        ),
        "distillation": pa.array(
            [json_dumps(dict(record.distillation), ensure_ascii=False, sort_keys=True) for record in records],
            type=pa.large_string(),
        ),
        "decontamination": pa.array(
            [json_dumps(dict(record.decontamination), ensure_ascii=False, sort_keys=True) for record in records],
            type=pa.large_string(),
        ),
        "meta": pa.array(
            [json_dumps(dict(record.meta), ensure_ascii=False, sort_keys=True) for record in records],
            type=pa.large_string(),
        ),
        "trace": pa.array(
            [json_dumps([event.to_dict() for event in record.trace], ensure_ascii=False, sort_keys=True) for record in records],
            type=pa.large_string(),
        ),
    }
    return pa.table(arrays, schema=CANONICAL_RECORD_SCHEMA)


def write_canonical_records(records: Iterable[CanonicalRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 200_000
    writer: pq.ParquetWriter | None = None
    buffered: list[CanonicalRecord] = []
    wrote_any = False
    try:
        for record in records:
            buffered.append(record)
            if len(buffered) < chunk_size:
                continue
            table = _records_to_table(buffered)
            if writer is None:
                writer = pq.ParquetWriter(target, CANONICAL_RECORD_SCHEMA, compression="snappy")
            writer.write_table(table)
            wrote_any = True
            buffered = []
        if buffered:
            table = _records_to_table(buffered)
            if writer is None:
                writer = pq.ParquetWriter(target, CANONICAL_RECORD_SCHEMA, compression="snappy")
            writer.write_table(table)
            wrote_any = True
    finally:
        if writer is not None:
            writer.close()
    if not wrote_any:
        pq.write_table(_empty_table(CANONICAL_RECORD_SCHEMA), target, compression="snappy")


def read_canonical_records(path: str | Path, *, columns: list[str] | None = None) -> list[CanonicalRecord]:
    target = Path(path)
    if target.is_dir():
        dataset = ds.dataset(target, format="parquet")
        table = dataset.to_table(columns=columns, use_threads=True)
    else:
        table = pq.read_table(target, columns=columns)
    return CanonicalRecord.from_table(table)


def read_canonical_records_parallel(
    path: str | Path,
    *,
    workers: int = 1,
    columns: list[str] | None = None,
    progress_every: int = 0,
    progress_callback: Callable[[int, int, int, float], None] | None = None,
) -> list[CanonicalRecord]:
    target = Path(path)
    if not target.is_dir() or workers <= 1:
        return read_canonical_records(target, columns=columns)

    dataset = ds.dataset(target, format="parquet")
    fragments = sorted(dataset.get_fragments(), key=lambda fragment: fragment.path)
    if not fragments:
        return []
    records: list[CanonicalRecord] = []
    started_at = time.perf_counter()

    def _read_one_fragment(fragment: ds.Fragment) -> list[CanonicalRecord]:
        table = fragment.to_table(columns=columns, use_threads=True)
        return CanonicalRecord.from_table(table)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(_read_one_fragment, fragments, chunksize=1)
        for completed, shard_records in enumerate(iterator, start=1):
            records.extend(shard_records)
            if progress_callback and progress_every > 0 and (
                completed % progress_every == 0 or completed == len(fragments)
            ):
                progress_callback(completed, len(fragments), len(records), time.perf_counter() - started_at)
    return records


def filter_canonical_records_to_output(
    input_path: str | Path,
    output_path: str | Path,
    *,
    keep_ids: set[str],
    workers: int = 1,
    unique_by_record_id: bool = False,
    progress_every: int = 0,
    progress_callback: Callable[[int, int, int, float], None] | None = None,
) -> int:
    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(source if source.is_dir() else [source], format="parquet")
    fragments = sorted(dataset.get_fragments(), key=lambda fragment: fragment.path)
    if not fragments:
        pq.write_table(_empty_table(CANONICAL_RECORD_SCHEMA), target, compression="snappy")
        return 0
    keep_values = pa.array(sorted(keep_ids))
    keep_set = pc.SetLookupOptions(value_set=keep_values)
    started_at = time.perf_counter()
    written_rows = 0
    writer: pq.ParquetWriter | None = None
    seen_output_ids: set[str] = set()

    def _filter_fragment(fragment: ds.Fragment) -> pa.Table:
        table = fragment.to_table(use_threads=True)
        mask = pc.is_in(table.column("record_id"), options=keep_set)
        filtered = table.filter(mask)
        if filtered.num_rows:
            filtered = filtered.cast(CANONICAL_RECORD_SCHEMA)
        return filtered

    def _dedup(table: pa.Table) -> pa.Table:
        if not unique_by_record_id or table.num_rows == 0:
            return table
        kept_rows: list[dict[str, object]] = []
        for row in table.to_pylist():
            record_id = str(row.get("record_id", ""))
            if record_id in seen_output_ids:
                continue
            seen_output_ids.add(record_id)
            kept_rows.append(row)
        if not kept_rows:
            return _empty_table(CANONICAL_RECORD_SCHEMA)
        return pa.Table.from_pylist(kept_rows, schema=CANONICAL_RECORD_SCHEMA)

    try:
        if workers > 1 and len(fragments) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                iterator = executor.map(_filter_fragment, fragments, chunksize=1)
                for index, filtered in enumerate(iterator, start=1):
                    filtered = _dedup(filtered)
                    if filtered.num_rows:
                        if writer is None:
                            writer = pq.ParquetWriter(target, CANONICAL_RECORD_SCHEMA, compression="snappy")
                        writer.write_table(filtered)
                        written_rows += filtered.num_rows
                    if progress_callback and progress_every > 0 and (
                        index % progress_every == 0 or index == len(fragments)
                    ):
                        progress_callback(index, len(fragments), written_rows, time.perf_counter() - started_at)
        else:
            for index, fragment in enumerate(fragments, start=1):
                filtered = _dedup(_filter_fragment(fragment))
                if filtered.num_rows:
                    if writer is None:
                        writer = pq.ParquetWriter(target, CANONICAL_RECORD_SCHEMA, compression="snappy")
                    writer.write_table(filtered)
                    written_rows += filtered.num_rows
                if progress_callback and progress_every > 0 and (
                    index % progress_every == 0 or index == len(fragments)
                ):
                    progress_callback(index, len(fragments), written_rows, time.perf_counter() - started_at)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pq.write_table(_empty_table(CANONICAL_RECORD_SCHEMA), target, compression="snappy")
    return written_rows


def write_canonical_record_shards(
    record_rows: Iterable[dict[str, object]],
    output_dir: str | Path,
    *,
    shard_prefix: str = "part",
    target_shard_bytes: int = 200 * 1024 * 1024,
) -> list[Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in target_dir.glob(f"{shard_prefix}-*.parquet"):
        stale_path.unlink()
    shard_index = 0
    shard_paths: list[Path] = []
    buffered_rows: list[dict[str, object]] = []
    buffered_bytes = 0

    def flush() -> None:
        nonlocal shard_index, buffered_rows, buffered_bytes
        if not buffered_rows:
            return
        shard_path = target_dir / f"{shard_prefix}-{shard_index:05d}.parquet"
        table = _table_from_rows(buffered_rows, CANONICAL_RECORD_SCHEMA)
        pq.write_table(table, shard_path, compression="snappy")
        shard_paths.append(shard_path)
        shard_index += 1
        buffered_rows = []
        buffered_bytes = 0

    for row in record_rows:
        buffered_rows.append(row)
        buffered_bytes += len(json_dumps(row, ensure_ascii=False))
        if buffered_bytes >= target_shard_bytes:
            flush()
    flush()
    return shard_paths


def write_processor_results(results: Iterable[ProcessorResult], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "record_id": result.record.record_id,
            "keep": result.keep,
            "stage": result.stage,
            "processor": result.processor,
            "reason_code": result.reason_code,
            "details": json_dumps(result.details, ensure_ascii=False, sort_keys=True),
        }
        for result in results
    ]
    pq.write_table(_table_from_rows(rows, PROCESSOR_RESULT_SCHEMA), target, compression="snappy")


def write_dedup_pairs(pairs: Iterable[DedupPair], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    chunk_size = 50_000
    buffered: list[dict[str, object]] = []
    wrote_any = False
    try:
        for pair in pairs:
            buffered.append(pair.to_storage_dict())
            if len(buffered) < chunk_size:
                continue
            table = _table_from_rows(buffered, DEDUP_PAIR_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(target, DEDUP_PAIR_SCHEMA, compression="snappy")
            writer.write_table(table)
            buffered = []
            wrote_any = True
        if buffered:
            table = _table_from_rows(buffered, DEDUP_PAIR_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(target, DEDUP_PAIR_SCHEMA, compression="snappy")
            writer.write_table(table)
            wrote_any = True
    finally:
        if writer is not None:
            writer.close()
    if not wrote_any:
        pq.write_table(_empty_table(DEDUP_PAIR_SCHEMA), target, compression="snappy")


def write_dedup_judgements(judgements: Iterable[DedupJudgement], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [judgement.to_storage_dict() for judgement in judgements]
    pq.write_table(_table_from_rows(rows, DEDUP_JUDGEMENT_SCHEMA), target, compression="snappy")


def write_rollout_results(results: Iterable[RolloutResult], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.to_storage_dict() for result in results]
    pq.write_table(_table_from_rows(rows, ROLLOUT_RESULT_SCHEMA), target, compression="snappy")
