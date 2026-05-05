from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

from pipeline.core.io import (
    discover_layout,
    load_yaml,
    read_canonical_records_parallel,
    write_canonical_records,
    write_dedup_judgements,
    write_dedup_pairs,
    write_processor_results,
)
from pipeline.core.incremental import run_incremental_add, run_merge_records
from pipeline.core.pipeline import run_pipeline
from pipeline.core.registry import build_processor, list_processors
from pipeline.core.schema import (
    CanonicalRecord,
    DatasetProcessor,
    DatasetProcessorResult,
    ProcessorResult,
    RecordProcessor,
)
from pipeline.ops.dedup._minhash_engine import VerifyResult
from pipeline.ops.llm.backends import build_inference_client, load_backend_config
from pipeline.ops.llm.client import BackendConfig, InferenceRequest
from pipeline.sources import build_source_adapter
from pipeline.sources.base import SourceConfig


log = logging.getLogger(__name__)


def _clean_incomplete_rollout_samples(
    records: list[CanonicalRecord],
    *,
    min_samples: int = 8,
) -> tuple[list[CanonicalRecord], int]:
    """Remove incomplete rollout verification to force re-processing.

    A record's rollout is considered complete only if it has at least
    ``min_samples`` entries in ``verification.samples``.  Records with
    fewer (including zero / api_error) get their verification cleared so
    the rollout processor will pick them up again.

    Returns (cleaned_records, num_cleaned).
    """
    cleaned = []
    num_cleaned = 0
    no_verification = 0

    for record in records:
        v = record.verification
        if not v:
            cleaned.append(record)
            no_verification += 1
            continue

        samples = v.get("samples")
        n = v.get("num_samples", 0)

        if samples and len(samples) >= min_samples and n >= min_samples:
            # Complete — keep as-is
            cleaned.append(record)
        elif samples or n > 0:
            # Incomplete — clear rollout fields to force retry
            num_cleaned += 1
            new_v = {
                k: val for k, val in v.items()
                if k not in {
                    "samples", "num_samples", "majority_answer",
                    "majority_count", "majority_matches_gt", "pass_ratio",
                    "model", "fallback_model", "used_fallback",
                    "raw_dataset_answer",
                }
            }
            cleaned.append(record.clone(verification=new_v if new_v else {}))
        else:
            cleaned.append(record)

    complete = len(records) - num_cleaned - no_verification
    log.info(
        "Rollout cleanup: total=%d complete=%d incomplete=%d no_verification=%d",
        len(records), complete, num_cleaned, no_verification,
    )
    return cleaned, num_cleaned


_STAGE_NAME_MAP: dict[str, list[dict[str, Any]]] = {
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
    "stage1_classify_v2": [
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


def _load_backend_client(config_path: str | None):
    if not config_path:
        return None, None
    config = load_backend_config(config_path)
    return build_inference_client(config), config


def _expand_stage_name(name: str, cli_args: argparse.Namespace) -> list[dict[str, Any]]:
    """Convert a stage name or processor name into a list of processor configs."""
    if name in _STAGE_NAME_MAP:
        warnings.warn(
            f"Stage name '{name}' is deprecated. Use individual processor names instead "
            f"(e.g. {', '.join(p['name'] for p in _STAGE_NAME_MAP[name])}).",
            DeprecationWarning,
            stacklevel=3,
        )
        base_processors = _STAGE_NAME_MAP[name]
    else:
        base_processors = [{"name": name}]

    result: list[dict[str, Any]] = []
    for proc in base_processors:
        cfg = dict(proc)
        cfg["enabled"] = True
        if hasattr(cli_args, "threshold"):
            cfg.setdefault("threshold", cli_args.threshold)
        if hasattr(cli_args, "dedup_num_perm"):
            cfg.setdefault("num_perm", cli_args.dedup_num_perm)
        if hasattr(cli_args, "dedup_bands"):
            cfg.setdefault("bands", cli_args.dedup_bands)
        if hasattr(cli_args, "dedup_ngram_size"):
            cfg.setdefault("ngram_size", cli_args.dedup_ngram_size)
        if hasattr(cli_args, "dedup_tokenizer"):
            cfg.setdefault("tokenizer", cli_args.dedup_tokenizer)
        if hasattr(cli_args, "dedup_preprocess_text"):
            cfg.setdefault("preprocess_text", bool(cli_args.dedup_preprocess_text))
        if hasattr(cli_args, "workers"):
            cfg.setdefault("workers", max(int(cli_args.workers), 1))
        if hasattr(cli_args, "dedup_max_candidates_per_record"):
            cfg.setdefault("max_candidates_per_record", max(int(cli_args.dedup_max_candidates_per_record), 0))
        if hasattr(cli_args, "min_ratio"):
            cfg.setdefault("min_ratio", cli_args.min_ratio)
        if hasattr(cli_args, "max_ratio"):
            cfg.setdefault("max_ratio", cli_args.max_ratio)
        if hasattr(cli_args, "model_config") and cli_args.model_config:
            cfg.setdefault("model_config", cli_args.model_config)
        result.append(cfg)
    return result


def command_ingest(args: argparse.Namespace) -> int:
    source_config = SourceConfig.from_dict(load_yaml(args.config))
    adapter = build_source_adapter(source_config)
    summary = adapter.write()
    print(f"ingested_records={summary.record_count}")
    return 0


def command_run_stage(args: argparse.Namespace) -> int:
    layout = discover_layout(Path(args.project_root))
    layout.ensure()
    stage_name = args.processor or args.stage
    if not stage_name:
        raise ValueError("run-stage requires --stage or --processor")
    output_path = Path(args.output)

    def _read_progress(completed: int, total: int, loaded_records: int, elapsed_sec: float) -> None:
        rate = (loaded_records / elapsed_sec) if elapsed_sec > 0 else 0.0
        print(
            f"[read] shards={completed}/{total} records={loaded_records} "
            f"elapsed_sec={elapsed_sec:.1f} records_per_sec={rate:.1f}",
            flush=True,
        )

    records = read_canonical_records_parallel(
        args.input,
        workers=max(int(args.workers), 1),
        progress_every=10,
        progress_callback=_read_progress,
    )

    processor_configs = _expand_stage_name(stage_name, args)

    config: dict[str, Any] = {"processors": processor_configs}
    result = run_pipeline(records, config, workers=max(int(args.workers), 1))
    write_canonical_records(result.records, output_path)

    if args.judgements_path and result.processor_results:
        write_processor_results(result.processor_results, args.judgements_path)

    if args.pairs_path:
        def _iter_pairs():
            for artifact_dict in result.artifacts.values():
                if not isinstance(artifact_dict, dict):
                    continue
                pairs = artifact_dict.get("candidate_pairs", [])
                if isinstance(pairs, VerifyResult):
                    for item in pairs.to_dedup_pairs():
                        yield item
                else:
                    for item in pairs:
                        yield item
                exact_dups = artifact_dict.get("exact_duplicates", [])
                for item in exact_dups:
                    yield item

        write_dedup_pairs(_iter_pairs(), args.pairs_path)

    if args.judgements_path:
        all_judgements = []
        for artifact_dict in result.artifacts.values():
            if isinstance(artifact_dict, dict):
                judgements = artifact_dict.get("judgements", [])
                if judgements:
                    all_judgements.extend(judgements)
        if all_judgements:
            write_dedup_judgements(all_judgements, args.judgements_path)

    print(f"output_records={len(result.records)}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    from pipeline.core.checkpoint import Checkpoint

    config = load_yaml(args.config)
    workers = max(int(args.workers or config.get("workers", 1)), 1)
    checkpoint_dir = config.get("checkpoint_dir")

    # --- Clean incomplete rollout samples from checkpoint ---
    # The checkpoint stores results from the previous run. If the previous run
    # had failures (api_error, incomplete samples), those records are stored
    # in the checkpoint with partial/empty verification. The rollout processor
    # skips records that already have verification.samples, so we need to:
    # 1. Load checkpoint records (not input records — input has no verification)
    # 2. Clear verification on incomplete records
    # 3. Invalidate the checkpoint so the pipeline re-runs the processor
    if checkpoint_dir:
        ckpt = Checkpoint(checkpoint_dir)
        rollout_steps = [
            p["name"] for p in config.get("processors", [])
            if isinstance(p, dict) and p.get("name", "").startswith("rollout_")
        ]
        # Get min_samples from processor config (num_samples field), default 8
        rollout_cfg = next(
            (p for p in config.get("processors", [])
             if isinstance(p, dict) and p.get("name", "").startswith("rollout_")),
            {},
        )
        min_samples = int(rollout_cfg.get("num_samples", 8))

        for step_name in rollout_steps:
            if not ckpt.is_completed(step_name):
                continue
            try:
                ckpt_records = ckpt.load_output(step_name)
            except FileNotFoundError:
                continue
            cleaned_records, num_cleaned = _clean_incomplete_rollout_samples(
                ckpt_records, min_samples=min_samples
            )

            if num_cleaned > 0:
                log.info(
                    "Cleaned %d incomplete rollout samples from checkpoint '%s' (min_samples=%d). "
                    "Invalidating checkpoint to force re-processing.",
                    num_cleaned, step_name, min_samples,
                )
                # Write cleaned records as the new input (preserving completed ones)
                from pipeline.core.io import write_canonical_records as _write
                cleaned_input = Path(config["input_path"]).with_name(
                    Path(config["input_path"]).stem + "_retry.parquet"
                )
                _write(cleaned_records, cleaned_input)
                config["input_path"] = str(cleaned_input)
                log.info("Wrote cleaned records to %s (%d records)", cleaned_input, len(cleaned_records))

                # Invalidate checkpoint: remove step from completed list
                index = ckpt._index()
                completed = [s for s in index.get("completed", []) if s != step_name]
                index["completed"] = completed
                ckpt._write_index(index)
                log.info("Checkpoint '%s' invalidated.", step_name)
            else:
                log.info(
                    "Checkpoint '%s': all %d records complete (min_samples=%d). Skipping.",
                    step_name, len(ckpt_records), min_samples,
                )

    records = read_canonical_records_parallel(config["input_path"], workers=workers)
    result = run_pipeline(records, config, checkpoint_dir=checkpoint_dir, workers=workers)
    write_canonical_records(result.records, config["output_path"])
    print(f"final_output_path={config['output_path']}")
    print(f"final_records={len(result.records)}")
    return 0


def command_add_dataset(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    config["__config_path"] = str(Path(args.config).resolve())
    workers = max(int(args.workers or config.get("workers", 1)), 1)
    result = run_incremental_add(config, workers=workers)
    print(f"phase1_records={result.phase1_records}")
    print(f"combined_records={result.combined_records}")
    print(f"new_survivors={result.new_survivors}")
    print(f"merged_total={result.merged_total}")
    print(f"output_path={result.output_path}")
    return 0


def command_merge_records(args: argparse.Namespace) -> int:
    config = {
        "existing_data_path": args.existing,
        "new_data_path": args.new,
        "output_path": args.output,
    }
    workers = max(int(args.workers), 1)
    result = run_merge_records(config, workers=workers)
    print(f"existing_records={result['existing_records']}")
    print(f"new_records={result['new_records']}")
    print(f"merged_total={result['merged_total']}")
    print(f"output_path={result['output_path']}")
    return 0


def command_stats(args: argparse.Namespace) -> int:
    records = read_canonical_records_parallel(
        args.input,
        workers=max(int(args.workers), 1),
        columns=["dataset_name", "distillation"],
    )
    with_distillation = sum(1 for record in records if bool(record.distillation))
    dataset_counts: dict[str, int] = {}
    for record in records:
        dataset_counts[record.dataset_name] = dataset_counts.get(record.dataset_name, 0) + 1
    print(json.dumps({"records": len(records), "with_distillation": with_distillation, "datasets": dataset_counts}, ensure_ascii=False, indent=2))
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    records = read_canonical_records_parallel(args.input, workers=max(int(args.workers), 1))
    payload = [record.to_dict() for record in records[: args.limit]]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    records = read_canonical_records_parallel(
        args.input,
        workers=max(int(args.workers), 1),
        columns=["record_id", "question", "dataset_name"],
    )
    errors: list[str] = []
    for index, record in enumerate(records):
        if not record.record_id:
            errors.append(f"row {index}: missing record_id")
        if not record.question:
            errors.append(f"row {index}: missing question")
        if not record.dataset_name:
            errors.append(f"row {index}: missing dataset_name")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"validated_records={len(records)}")
    return 0


def command_backend_probe(args: argparse.Namespace) -> int:
    client, backend = _load_backend_client(args.config)
    if client is None or backend is None:
        raise ValueError("Backend config is required")
    response = client.infer(
        InferenceRequest(
            model=backend.model,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
    )
    print(response.text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--config", required=True)
    ingest_parser.set_defaults(func=command_ingest)

    run_stage_parser = subparsers.add_parser("run-stage")
    run_stage_parser.add_argument("--project-root", default=".")
    run_stage_parser.add_argument("--stage")
    run_stage_parser.add_argument("--processor")
    run_stage_parser.add_argument("--input", required=True)
    run_stage_parser.add_argument("--output", required=True)
    run_stage_parser.add_argument("--model-config")
    run_stage_parser.add_argument("--pairs-path")
    run_stage_parser.add_argument("--judgements-path")
    run_stage_parser.add_argument("--threshold", type=float, default=0.8)
    run_stage_parser.add_argument("--min-ratio", type=float, default=0.0)
    run_stage_parser.add_argument("--max-ratio", type=float, default=1.0)
    run_stage_parser.add_argument("--workers", type=int, default=1)
    run_stage_parser.add_argument("--stage1-processor-limit", type=int, default=0)
    run_stage_parser.add_argument("--dedup-num-perm", type=int, default=64)
    run_stage_parser.add_argument("--dedup-bands", type=int, default=8)
    run_stage_parser.add_argument("--dedup-ngram-size", type=int, default=3)
    run_stage_parser.add_argument("--dedup-tokenizer", default="char")
    run_stage_parser.add_argument("--dedup-preprocess-text", action=argparse.BooleanOptionalAction, default=True)
    run_stage_parser.add_argument("--dedup-max-candidates-per-record", type=int, default=200)
    run_stage_parser.set_defaults(func=command_run_stage)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--project-root", default=".")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.set_defaults(func=command_run)

    add_dataset_parser = subparsers.add_parser("add-dataset")
    add_dataset_parser.add_argument("--config", required=True)
    add_dataset_parser.add_argument("--workers", type=int)
    add_dataset_parser.set_defaults(func=command_add_dataset)

    merge_parser = subparsers.add_parser("merge-records")
    merge_parser.add_argument("--existing", required=True, help="Existing processed corpus parquet")
    merge_parser.add_argument("--new", required=True, help="New records parquet (e.g. after LLM classification)")
    merge_parser.add_argument("--output", required=True, help="Merged output parquet path")
    merge_parser.add_argument("--workers", type=int, default=1)
    merge_parser.set_defaults(func=command_merge_records)

    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("--input", required=True)
    stats_parser.add_argument("--workers", type=int, default=1)
    stats_parser.set_defaults(func=command_stats)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--input", required=True)
    inspect_parser.add_argument("--limit", type=int, default=3)
    inspect_parser.add_argument("--workers", type=int, default=1)
    inspect_parser.set_defaults(func=command_inspect)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--input", required=True)
    validate_parser.add_argument("--workers", type=int, default=1)
    validate_parser.set_defaults(func=command_validate)

    backend_parser = subparsers.add_parser("backend")
    backend_subparsers = backend_parser.add_subparsers(dest="backend_command", required=True)
    probe_parser = backend_subparsers.add_parser("probe")
    probe_parser.add_argument("--config", required=True)
    probe_parser.add_argument("--prompt", default="Respond with the single word pong.")
    probe_parser.add_argument("--system-prompt", default="")
    probe_parser.add_argument("--max-tokens", type=int, default=32)
    probe_parser.set_defaults(func=command_backend_probe)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run-stage" and not getattr(args, "stage", None) and not getattr(args, "processor", None):
        parser.error("run-stage requires --stage or --processor")
    return int(args.func(args))
