from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.core.checkpoint import Checkpoint
from pipeline.core.io import load_yaml, write_canonical_records
from pipeline.core.pipeline import run_pipeline
from pipeline.core.schema import CanonicalRecord

log = logging.getLogger(__name__)

DEDUP_PROCESSOR_NAMES = {"exact_dedup", "fuzzy_dedup"}


@dataclass(slots=True)
class IncrementalAddResult:
    phase1_records: int
    combined_records: int
    new_survivors: int
    merged_total: int
    output_path: str


def _enabled_processors(processors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in processors if isinstance(item, dict) and item.get("enabled", True)]


def split_processors_for_incremental(
    processors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    enabled = _enabled_processors(processors)
    dedup_indexes = [index for index, item in enumerate(enabled) if str(item.get("name", "")) in DEDUP_PROCESSOR_NAMES]
    if not dedup_indexes:
        raise ValueError("Incremental add requires at least one dedup processor (exact_dedup or fuzzy_dedup).")

    last_dedup = dedup_indexes[-1]
    phase1: list[dict[str, Any]] = []
    phase2: list[dict[str, Any]] = []
    for item in enabled[: last_dedup + 1]:
        if str(item.get("name", "")) in DEDUP_PROCESSOR_NAMES:
            phase2.append(dict(item))
        else:
            phase1.append(dict(item))
    phase3 = enabled[last_dedup + 1 :]
    return phase1, phase2, phase3


def augment_priority_config(
    processor_config: dict[str, Any],
    new_tier: int,
    new_names: list[str],
) -> dict[str, Any]:
    updated = dict(processor_config)
    raw_priority = updated.get("dataset_priority") or {}
    priority = {int(key): [str(name) for name in value] for key, value in dict(raw_priority).items()}
    names = [str(name) for name in new_names if str(name).strip()]
    if not names:
        raise ValueError("new_dataset_names must contain at least one dataset name.")
    if new_tier in priority:
        merged = list(priority[new_tier])
        for name in names:
            if name not in merged:
                merged.append(name)
        priority[new_tier] = merged
    else:
        priority[new_tier] = names
    updated["dataset_priority"] = priority
    return updated


def _resolve_path(config: dict[str, Any], key: str) -> Path:
    value = str(config.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing required config field: {key}")
    candidate = Path(value)
    config_path = str(config.get("__config_path", "")).strip()
    if candidate.is_absolute() or not config_path:
        return candidate
    return (Path(config_path).resolve().parent / candidate).resolve()


def _ensure_list_of_strings(config: dict[str, Any], key: str) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    return [str(item) for item in value if str(item).strip()]


def _filter_records_by_dataset_names(records: list[CanonicalRecord], dataset_names: set[str]) -> list[CanonicalRecord]:
    return [record for record in records if record.dataset_name in dataset_names]


def _merge_records(existing_records: list[CanonicalRecord], new_records: list[CanonicalRecord]) -> list[CanonicalRecord]:
    merged = list(existing_records)
    seen_ids = {record.record_id for record in existing_records}
    for record in new_records:
        if record.record_id in seen_ids:
            continue
        merged.append(record)
        seen_ids.add(record.record_id)
    return merged


# ---------------------------------------------------------------------------
# Pre-tag rules: tag records based on fields in the original source parquet
# ---------------------------------------------------------------------------

def _build_pre_tag_uid_set(
    source_parquet: Path,
    field: str,
    match_value: str,
) -> set[str]:
    """Read a source parquet and return UIDs where *field* equals *match_value*.

    *field* supports dotted paths for nested structs (e.g. ``meta_data.domain``).
    """
    import pyarrow.dataset as ds

    top_col = field.split(".")[0]
    parts = field.split(".")
    dataset = ds.dataset(source_parquet, format="parquet")
    matching: set[str] = set()
    for batch in dataset.scanner(columns=["uid", top_col], batch_size=50000).to_batches():
        for row in batch.to_pylist():
            val: Any = row
            for part in parts:
                if isinstance(val, dict):
                    val = val.get(part)
                else:
                    val = None
                    break
            if str(val) == match_value:
                uid = str(row.get("uid", ""))
                if uid:
                    matching.add(uid)
    return matching


def _apply_pre_tag_rules(
    records: list[CanonicalRecord],
    rules: list[dict[str, Any]],
    config_dir: Path | None,
) -> int:
    """Apply pre-tag rules to *records* in-place. Returns count of tagged records."""
    if not rules:
        return 0

    total_tagged = 0
    for rule in rules:
        raw_path = str(rule.get("source_parquet", "")).strip()
        if not raw_path:
            continue
        source = Path(raw_path)
        if not source.is_absolute() and config_dir:
            source = (config_dir / source).resolve()

        field = str(rule.get("field", "")).strip()
        match_value = str(rule.get("match_value", "")).strip()
        training_phase = str(rule.get("training_phase", "midtrain"))
        filter_tag = str(rule.get("filter_tag", ""))

        if not field or not match_value:
            log.warning("pre_tag_rules: skipping rule with missing field or match_value")
            continue

        log.info("pre_tag: reading %s for field=%s match=%s", source, field, match_value)
        matching_uids = _build_pre_tag_uid_set(source, field, match_value)
        if not matching_uids:
            log.info("pre_tag: no matching UIDs found, skipping")
            continue

        tagged = 0
        clone_kwargs: dict[str, str] = {"training_phase": training_phase}
        if filter_tag:
            clone_kwargs["filter_tag"] = filter_tag
        for i, rec in enumerate(records):
            if rec.record_id in matching_uids:
                records[i] = rec.clone(**clone_kwargs)
                records[i].add_trace(
                    stage="pre_tag",
                    processor="pre_tag_rules",
                    status="tagged",
                    details={"field": field, "match_value": match_value, "training_phase": training_phase},
                )
                tagged += 1
        total_tagged += tagged
        log.info(
            "pre_tag: field=%s match=%s → %s | matching_uids=%d tagged_records=%d",
            field, match_value, training_phase, len(matching_uids), tagged,
        )

    return total_tagged


def run_incremental_add(config: dict[str, Any], *, workers: int) -> IncrementalAddResult:
    pipeline_config_path = _resolve_path(config, "pipeline_config")
    new_data_path = _resolve_path(config, "new_data_path")
    existing_data_path = _resolve_path(config, "existing_data_path")
    output_path = _resolve_path(config, "output_path")
    checkpoint_dir = _resolve_path(config, "checkpoint_dir")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    skip_merge = bool(config.get("skip_merge", False))

    pipeline_config = load_yaml(pipeline_config_path)
    processors = pipeline_config.get("processors")
    if not isinstance(processors, list):
        raise ValueError(f"Pipeline config {pipeline_config_path} must contain a processors list.")

    new_dataset_names = _ensure_list_of_strings(config, "new_dataset_names")
    new_dataset_name_set = set(new_dataset_names)
    new_dataset_priority_tier = int(config.get("new_dataset_priority_tier", 0))
    if new_dataset_priority_tier < 1:
        raise ValueError("new_dataset_priority_tier must be >= 1")

    phase1_processors, phase2_processors, phase3_processors = split_processors_for_incremental(processors)
    phase2_processors = [
        augment_priority_config(item, new_dataset_priority_tier, new_dataset_names) for item in phase2_processors
    ]

    phase_boundary_checkpoint = Checkpoint(checkpoint_dir)

    if phase_boundary_checkpoint.is_completed("phase1_clean"):
        phase1_records = phase_boundary_checkpoint.load_output("phase1_clean")
    else:
        new_records = read_canonical_records_parallel_safe(new_data_path, workers=workers)

        pre_tag_rules = config.get("pre_tag_rules")
        if isinstance(pre_tag_rules, list) and pre_tag_rules:
            config_path = str(config.get("__config_path", "")).strip()
            config_dir = Path(config_path).resolve().parent if config_path else None
            pre_tagged = _apply_pre_tag_rules(new_records, pre_tag_rules, config_dir)
            log.info("pre-tagged %d / %d new records", pre_tagged, len(new_records))

        phase1_result = run_pipeline(
            new_records,
            {"processors": phase1_processors},
            checkpoint_dir=checkpoint_dir / "phase1_pipeline",
            workers=workers,
        )
        phase1_records = phase1_result.records
        phase_boundary_checkpoint.save_output("phase1_clean", phase1_records)

    existing_records = read_canonical_records_parallel_safe(existing_data_path, workers=workers)
    combined_records = _merge_records(existing_records, phase1_records)

    if phase_boundary_checkpoint.is_completed("phase2_new_survivors"):
        new_survivors = phase_boundary_checkpoint.load_output("phase2_new_survivors")
    else:
        phase2_result = run_pipeline(
            combined_records,
            {"processors": phase2_processors},
            checkpoint_dir=checkpoint_dir / "phase2_pipeline",
            workers=workers,
        )
        kept_ids = {record.record_id for record in phase2_result.records}
        new_survivors = [
            record for record in phase1_records if record.record_id in kept_ids and record.dataset_name in new_dataset_name_set
        ]
        phase_boundary_checkpoint.save_output("phase2_new_survivors", new_survivors)

    if phase_boundary_checkpoint.is_completed("phase3_classify"):
        classified_new_records = phase_boundary_checkpoint.load_output("phase3_classify")
    else:
        phase3_result = run_pipeline(
            new_survivors,
            {"processors": phase3_processors},
            checkpoint_dir=checkpoint_dir / "phase3_pipeline",
            workers=workers,
        )
        classified_new_records = phase3_result.records
        phase_boundary_checkpoint.save_output("phase3_classify", classified_new_records)

    if skip_merge:
        write_canonical_records(classified_new_records, output_path)
        log.info(
            "skip_merge=true → wrote %d new records to %s (merge deferred for LLM classification)",
            len(classified_new_records), output_path,
        )
        return IncrementalAddResult(
            phase1_records=len(phase1_records),
            combined_records=len(combined_records),
            new_survivors=len(classified_new_records),
            merged_total=0,
            output_path=str(output_path),
        )

    merged_records = _merge_records(existing_records, classified_new_records)
    write_canonical_records(merged_records, output_path)

    return IncrementalAddResult(
        phase1_records=len(phase1_records),
        combined_records=len(combined_records),
        new_survivors=len(classified_new_records),
        merged_total=len(merged_records),
        output_path=str(output_path),
    )


def run_merge_records(config: dict[str, Any], *, workers: int) -> dict[str, Any]:
    """Merge existing records with new (LLM-classified) records into a single output."""
    existing_data_path = _resolve_path(config, "existing_data_path")
    new_data_path = _resolve_path(config, "new_data_path")
    output_path = _resolve_path(config, "output_path")

    log.info("merge: loading existing from %s", existing_data_path)
    existing_records = read_canonical_records_parallel_safe(existing_data_path, workers=workers)
    log.info("merge: loading new from %s", new_data_path)
    new_records = read_canonical_records_parallel_safe(new_data_path, workers=workers)

    merged = _merge_records(existing_records, new_records)
    write_canonical_records(merged, output_path)

    log.info(
        "merge: existing=%d new=%d merged=%d → %s",
        len(existing_records), len(new_records), len(merged), output_path,
    )
    return {
        "existing_records": len(existing_records),
        "new_records": len(new_records),
        "merged_total": len(merged),
        "output_path": str(output_path),
    }


def read_canonical_records_parallel_safe(path: str | Path, *, workers: int) -> list[CanonicalRecord]:
    from pipeline.core.io import read_canonical_records_parallel

    return read_canonical_records_parallel(path, workers=max(int(workers), 1))
