from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import logging
from multiprocessing import get_context
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.core.checkpoint import Checkpoint
from pipeline.core.io import load_yaml
from pipeline.core.registry import build_processor
from pipeline.core.schema import (
    CanonicalRecord,
    DatasetProcessor,
    DatasetProcessorResult,
    ProcessorResult,
    RecordProcessor,
)

log = logging.getLogger(__name__)


def _normalize_processors(config: dict[str, Any]) -> list[dict[str, Any]]:
    processors = config.get("processors")
    if isinstance(processors, list):
        return [dict(item) for item in processors if isinstance(item, dict)]
    stages = config.get("stages", [])
    if not isinstance(stages, list):
        return []

    stage_to_processors: dict[str, list[dict[str, str]]] = {
        "stage0_text_filter": [{"name": "math_filter"}],
        "stage1_non_cot": [
            {"name": "text_cleaning"},
            {"name": "language_filter"},
            {"name": "length_filter"},
            {"name": "mcq_converter"},
            {"name": "rule_correctness"},
        ],
        "stage1_classify": [
            {"name": "text_normalize"},
            {"name": "instruction_wrapper_cleaner"},
            {"name": "text_cleaning"},
            {"name": "prompt_contamination_filter"},
            {"name": "missing_context_filter"},
            {"name": "length_filter"},
            {"name": "language_filter"},
            {"name": "math_filter"},
            {"name": "phantom_reference_filter"},
            {"name": "broken_question_filter"},
            {"name": "mcq_converter"},
            {"name": "meta_task_filter"},
            {"name": "boolean_answer_filter"},
            {"name": "answer_quality_filter"},
        ],
        "stage2_dedup": [{"name": "exact_dedup"}, {"name": "fuzzy_dedup"}],
        "stage3_correctness": [{"name": "rollout_correctness"}, {"name": "difficulty_annotation"}],
        "stage4_cot": [{"name": "cot_cleaning"}],
        "stage5_pass_ratio": [{"name": "pass_ratio_filter"}],
    }

    normalized: list[dict[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, dict) or not stage.get("enabled", True):
            continue
        name = str(stage.get("name", ""))
        targets = stage_to_processors.get(name, [{"name": name}])
        for target in targets:
            payload = dict(stage)
            payload.update(target)
            normalized.append(payload)
    return normalized


def _apply_record_processors(
    records: list[CanonicalRecord],
    processors: list[RecordProcessor],
    *,
    workers: int = 1,
) -> tuple[list[CanonicalRecord], list[ProcessorResult]]:
    if workers <= 1 or len(records) <= 1:
        all_records: list[CanonicalRecord] = []
        results: list[ProcessorResult] = []
        for record in records:
            current = record
            for processor in processors:
                result = processor.process(current)
                results.append(result)
                current = result.record
            all_records.append(current)
        return all_records, results

    processor_specs = [{"name": processor.name, **processor.config} for processor in processors]
    chunk_size = max(1, len(records) // (workers * 4))
    chunks = [records[index : index + chunk_size] for index in range(0, len(records), chunk_size)]
    all_records: list[CanonicalRecord] = []
    results: list[ProcessorResult] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("fork"),
    ) as executor:
        tasks = ((chunk, processor_specs) for chunk in chunks)
        for chunk_records, chunk_results in executor.map(_apply_record_processors_worker, tasks, chunksize=1):
            all_records.extend(chunk_records)
            results.extend(chunk_results)
    return all_records, results


def _apply_record_processors_worker(
    args: tuple[list[CanonicalRecord], list[dict[str, Any]]],
) -> tuple[list[CanonicalRecord], list[ProcessorResult]]:
    import pipeline.ops  # noqa: F401

    chunk, processor_specs = args
    processors: list[RecordProcessor] = []
    for spec in processor_specs:
        instance = build_processor(str(spec["name"]), spec)
        if not isinstance(instance, RecordProcessor):
            raise TypeError(f"Processor {spec['name']} is not a RecordProcessor")
        processors.append(instance)

    all_records: list[CanonicalRecord] = []
    results: list[ProcessorResult] = []
    for record in chunk:
        current = record
        for processor in processors:
            result = processor.process(current)
            results.append(result)
            current = result.record
        all_records.append(current)
    return all_records, results


@dataclass(slots=True)
class PipelineRunResult:
    records: list[CanonicalRecord]
    artifacts: dict[str, Any]
    processor_results: list[ProcessorResult]


def run_pipeline(
    records: list[CanonicalRecord],
    config: dict[str, Any],
    *,
    checkpoint_dir: str | Path | None = None,
    workers: int = 1,
) -> PipelineRunResult:
    import pipeline.ops  # noqa: F401

    checkpoint = Checkpoint(checkpoint_dir) if checkpoint_dir else None
    processors = _normalize_processors(config)
    current = records
    all_processor_results: list[ProcessorResult] = []
    artifacts: dict[str, Any] = {}
    index = 0

    while index < len(processors):
        step = processors[index]
        if not step.get("enabled", True):
            index += 1
            continue
        name = str(step["name"])
        step_key = str(step.get("step_name", name))

        if checkpoint and checkpoint.is_completed(step_key):
            current = checkpoint.load_output(step_key)
            log.debug("checkpoint hit for %s, skipping", step_key)
            index += 1
            continue

        instance = build_processor(name, step)

        if isinstance(instance, RecordProcessor):
            group_keys: list[str] = [step_key]
            group: list[RecordProcessor] = [instance]
            j = index + 1
            while j < len(processors):
                next_step = processors[j]
                if not next_step.get("enabled", True):
                    j += 1
                    continue
                next_name = str(next_step["name"])
                next_key = str(next_step.get("step_name", next_name))
                if checkpoint and checkpoint.is_completed(next_key):
                    break
                next_instance = build_processor(next_name, next_step)
                if not isinstance(next_instance, RecordProcessor):
                    break
                group.append(next_instance)
                group_keys.append(next_key)
                j += 1

            group_workers = max(workers, *(max(int(step_config.get("workers", 1)), 1) for step_config in [step, *processors[index + 1 : j]]))
            current, group_results = _apply_record_processors(current, group, workers=group_workers)
            all_processor_results.extend(group_results)

            if checkpoint:
                batch_label = "+".join(group_keys)
                checkpoint.save_output(batch_label, current)
                for gk in group_keys:
                    checkpoint.mark_completed(gk)
            index = j
            continue

        dataset_result = instance.process(current, pipeline_artifacts=artifacts)
        if not isinstance(dataset_result, DatasetProcessorResult):
            raise TypeError(f"Dataset processor {name} did not return DatasetProcessorResult")
        current = dataset_result.kept_records
        all_processor_results.extend(dataset_result.processor_results)
        artifacts[step_key] = dict(dataset_result.artifacts)
        if checkpoint:
            checkpoint.save_output(step_key, current)
        index += 1

    return PipelineRunResult(records=current, artifacts=artifacts, processor_results=all_processor_results)


def run_pipeline_from_config_path(
    records: list[CanonicalRecord],
    config_path: str | Path,
    *,
    checkpoint_dir: str | Path | None = None,
    workers: int = 1,
) -> PipelineRunResult:
    config = load_yaml(config_path)
    checkpoint = checkpoint_dir if checkpoint_dir is not None else config.get("checkpoint_dir")
    configured_workers = max(int(config.get("workers", workers)), 1)
    return run_pipeline(records, config, checkpoint_dir=checkpoint, workers=configured_workers)
