from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.core.io import load_yaml
from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, DatasetProcessor, DatasetProcessorResult, ProcessorResult
from pipeline.ops.decontaminate.decontamination import (
    BenchmarkQuestion,
    JudgeResult,
    _base_decontamination_payload,
    _build_faiss_index,
    _build_json_schema,
    _extract_prompt_parts,
    _judge_candidates,
    _parse_judge_response,
    _truncate_text,
    load_benchmark_questions,
)
from pipeline.ops.decontaminate.embedding import embed_all, l2_normalize

log = logging.getLogger(__name__)

_BENCHMARKS_FILE = "benchmark_questions.json"
_LOAD_META_FILE = "load_meta.json"


def _resolve_paths(paths: list[str]) -> list[str]:
    resolved: list[str] = []
    for p in paths:
        target = Path(p)
        if target.exists():
            resolved.append(str(target))
        else:
            log.warning("benchmark path missing, skipped: %s", p)
    if not resolved:
        raise FileNotFoundError("no benchmark parquet files found")
    return resolved


def _safe_model_name(model: str) -> str:
    return model.replace("/", "_")


def _save_benchmarks(checkpoint_dir: str, benchmarks: list[BenchmarkQuestion]) -> Path:
    out = Path(checkpoint_dir) / _BENCHMARKS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "uid": b.uid,
            "dataset_name": b.dataset_name,
            "question": b.question,
            "ground_truth": b.ground_truth,
        }
        for b in benchmarks
    ]
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


def _load_benchmarks(checkpoint_dir: str) -> list[BenchmarkQuestion]:
    src = Path(checkpoint_dir) / _BENCHMARKS_FILE
    if not src.exists():
        raise FileNotFoundError(
            f"benchmark data not found at {src}. Run decontam_load first."
        )
    raw = json.loads(src.read_text(encoding="utf-8"))
    return [
        BenchmarkQuestion(
            uid=item["uid"],
            dataset_name=item["dataset_name"],
            question=item["question"],
            ground_truth=item["ground_truth"],
        )
        for item in raw
    ]


def _filter_target_records(
    records: list[CanonicalRecord],
    target_phases: set[str],
) -> tuple[list[int], list[str], list[str]]:
    """Return (indices, record_ids, questions) for target records."""
    indices: list[int] = []
    record_ids: list[str] = []
    questions: list[str] = []
    for idx, rec in enumerate(records):
        if rec.training_phase in target_phases:
            indices.append(idx)
            record_ids.append(rec.record_id)
            questions.append(rec.question or "")
    return indices, record_ids, questions


def _embedding_checkpoint_path(checkpoint_dir: str, prefix: str, model: str) -> str:
    return str(Path(checkpoint_dir) / f"{prefix}_{_safe_model_name(model)}.npy")


# ------------------------------------------------------------------
# Step 1: Load benchmark + training data, validate, save metadata
# ------------------------------------------------------------------

@register_processor("decontam_load")
class DecontamLoadProcessor(DatasetProcessor):
    name = "decontam_load"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.benchmark_paths = [str(p) for p in self.config.get("benchmark_paths", [])]
        self.target_phases = {str(v) for v in self.config.get("target_phases", ["posttrain", "midtrain"])}
        self.checkpoint_dir = str(self.config.get("checkpoint_dir", "")).strip()
        if not self.checkpoint_dir:
            raise ValueError("decontam_load requires checkpoint_dir")

    def process(
        self,
        records: list[CanonicalRecord],
        *,
        pipeline_artifacts: dict[str, Any] | None = None,
    ) -> DatasetProcessorResult:
        resolved = _resolve_paths(self.benchmark_paths)
        benchmarks = load_benchmark_questions(resolved)
        if not benchmarks:
            raise ValueError("no benchmark questions loaded")

        indices, record_ids, questions = _filter_target_records(records, self.target_phases)

        source_counts: dict[str, int] = Counter(b.dataset_name for b in benchmarks)
        phase_counts: dict[str, int] = Counter(r.training_phase for r in records)

        _save_benchmarks(self.checkpoint_dir, benchmarks)

        meta = {
            "total_records": len(records),
            "target_records": len(indices),
            "benchmark_questions": len(benchmarks),
            "benchmark_sources": dict(source_counts),
            "phase_distribution": dict(phase_counts),
            "target_phases": sorted(self.target_phases),
        }
        meta_path = Path(self.checkpoint_dir) / _LOAD_META_FILE
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        log.info(
            "decontam_load: total_records=%d target_records=%d benchmarks=%d sources=%s",
            len(records),
            len(indices),
            len(benchmarks),
            list(source_counts.keys()),
        )
        for source, cnt in sorted(source_counts.items()):
            log.info("  benchmark %-20s : %d questions", source, cnt)
        for phase, cnt in sorted(phase_counts.items(), key=lambda x: -x[1]):
            marker = " <-- target" if phase in self.target_phases else ""
            log.info("  phase %-20s : %d%s", repr(phase), cnt, marker)

        return DatasetProcessorResult(kept_records=records, artifacts=meta)


# ------------------------------------------------------------------
# Step 2: Embed training + benchmark questions, save checkpoints
# ------------------------------------------------------------------

@register_processor("decontam_embed")
class DecontamEmbedProcessor(DatasetProcessor):
    name = "decontam_embed"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.target_phases = {str(v) for v in self.config.get("target_phases", ["posttrain", "midtrain"])}
        self.checkpoint_dir = str(self.config.get("checkpoint_dir", "")).strip()
        if not self.checkpoint_dir:
            raise ValueError("decontam_embed requires checkpoint_dir")

        self.embedding_api_base = str(self.config.get("embedding_api_base", "http://localhost:39600/v1"))
        self.embedding_api_key = str(self.config.get("embedding_api_key", ""))
        self.embedding_model = str(self.config.get("embedding_model", "Qwen3-Embedding-4B"))
        self.embedding_batch_size = int(self.config.get("embedding_batch_size", 256))
        self.embedding_concurrency = int(self.config.get("embedding_concurrency", 32))
        self.embedding_timeout = int(self.config.get("embedding_timeout", 300))
        self.embedding_max_retries = int(self.config.get("embedding_max_retries", 3))
        self.encoding_format = str(self.config.get("encoding_format", "float"))

    def process(
        self,
        records: list[CanonicalRecord],
        *,
        pipeline_artifacts: dict[str, Any] | None = None,
    ) -> DatasetProcessorResult:
        benchmarks = _load_benchmarks(self.checkpoint_dir)
        _, record_ids, train_texts = _filter_target_records(records, self.target_phases)
        benchmark_texts = [b.question for b in benchmarks]

        log.info(
            "decontam_embed: train_texts=%d benchmark_texts=%d model=%s",
            len(train_texts),
            len(benchmark_texts),
            self.embedding_model,
        )

        train_ckpt = _embedding_checkpoint_path(self.checkpoint_dir, "train", self.embedding_model)
        bench_ckpt = _embedding_checkpoint_path(self.checkpoint_dir, "benchmark", self.embedding_model)

        train_emb = asyncio.run(
            embed_all(
                train_texts,
                api_base=self.embedding_api_base,
                model=self.embedding_model,
                batch_size=self.embedding_batch_size,
                concurrency=self.embedding_concurrency,
                api_key=self.embedding_api_key,
                timeout_seconds=self.embedding_timeout,
                max_retries=self.embedding_max_retries,
                encoding_format=self.encoding_format,
                checkpoint_path=train_ckpt,
                progress_name="train_embeddings",
            )
        )
        bench_emb = asyncio.run(
            embed_all(
                benchmark_texts,
                api_base=self.embedding_api_base,
                model=self.embedding_model,
                batch_size=self.embedding_batch_size,
                concurrency=self.embedding_concurrency,
                api_key=self.embedding_api_key,
                timeout_seconds=self.embedding_timeout,
                max_retries=self.embedding_max_retries,
                encoding_format=self.encoding_format,
                checkpoint_path=bench_ckpt,
                progress_name="benchmark_embeddings",
            )
        )

        log.info(
            "decontam_embed: train_shape=%s benchmark_shape=%s saved to %s",
            train_emb.shape,
            bench_emb.shape,
            self.checkpoint_dir,
        )

        return DatasetProcessorResult(
            kept_records=records,
            artifacts={
                "train_embedding_shape": list(train_emb.shape),
                "benchmark_embedding_shape": list(bench_emb.shape),
            },
        )


# ------------------------------------------------------------------
# Step 3: FAISS search + LLM judge, mark contaminated records
# ------------------------------------------------------------------

@register_processor("decontam_search")
class DecontamSearchProcessor(DatasetProcessor):
    name = "decontam_search"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.target_phases = {str(v) for v in self.config.get("target_phases", ["posttrain", "midtrain"])}
        self.checkpoint_dir = str(self.config.get("checkpoint_dir", "")).strip()
        if not self.checkpoint_dir:
            raise ValueError("decontam_search requires checkpoint_dir")

        self.embedding_model = str(self.config.get("embedding_model", "Qwen3-Embedding-4B"))
        self.top_k = int(self.config.get("top_k", 10))
        thr = self.config.get("similarity_threshold")
        self.similarity_threshold = None if thr in (None, "", "null") else float(thr)
        self.faiss_gpu_id = int(self.config.get("faiss_gpu_id", 0))

        self.llm_api_base = str(self.config.get("llm_api_base", "http://localhost:8000/v1"))
        self.llm_api_key = str(self.config.get("llm_api_key", ""))
        self.llm_model = str(self.config.get("llm_model", "gpt-oss-20b"))
        self.llm_concurrency = int(self.config.get("llm_concurrency", 256))
        self.llm_max_retries = int(self.config.get("llm_max_retries", 3))
        self.llm_api_timeout = int(self.config.get("llm_api_timeout", 300))
        self.llm_max_tokens = int(self.config.get("llm_max_tokens", 96))
        self.llm_temperature = float(self.config.get("llm_temperature", 0.0))
        self.llm_top_p = float(self.config.get("llm_top_p", 1.0))
        self.max_question_chars = int(self.config.get("max_question_chars", 4000))
        raw_llm_extra_body = self.config.get("llm_extra_body", {}) or {}
        self.llm_extra_body = dict(raw_llm_extra_body)

        prompt_path = str(self.config.get("prompt_path", "configs/prompts/decontamination_judge.yaml"))
        prompt_cfg = load_yaml(prompt_path)
        self.system_prompt, self.prompt_template = _extract_prompt_parts(prompt_cfg)

    def process(
        self,
        records: list[CanonicalRecord],
        *,
        pipeline_artifacts: dict[str, Any] | None = None,
    ) -> DatasetProcessorResult:
        benchmarks = _load_benchmarks(self.checkpoint_dir)
        target_indices, train_record_ids, train_texts = _filter_target_records(records, self.target_phases)

        train_ckpt = _embedding_checkpoint_path(self.checkpoint_dir, "train", self.embedding_model)
        bench_ckpt = _embedding_checkpoint_path(self.checkpoint_dir, "benchmark", self.embedding_model)
        for p in (train_ckpt, bench_ckpt):
            npy = Path(p)
            if not npy.exists() and not npy.with_suffix(".npy").exists():
                raise FileNotFoundError(f"embedding checkpoint not found: {p}. Run decontam_embed first.")

        from pipeline.ops.decontaminate.embedding import load_embedding_checkpoint

        train_emb = load_embedding_checkpoint(train_ckpt, model=self.embedding_model, num_texts=len(train_texts))
        if train_emb is None:
            raise FileNotFoundError(f"failed to load train embeddings from {train_ckpt}")
        bench_emb = load_embedding_checkpoint(bench_ckpt, model=self.embedding_model, num_texts=len(benchmarks))
        if bench_emb is None:
            raise FileNotFoundError(f"failed to load benchmark embeddings from {bench_ckpt}")

        train_emb = l2_normalize(train_emb)
        bench_emb = l2_normalize(bench_emb)

        log.info(
            "decontam_search: target=%d benchmarks=%d top_k=%d threshold=%s",
            len(target_indices), len(benchmarks), self.top_k, self.similarity_threshold,
        )

        index, gpu_resources = _build_faiss_index(train_emb, gpu_id=self.faiss_gpu_id)

        top_k = min(self.top_k, len(train_record_ids))
        distances, neighbor_indices = index.search(bench_emb, top_k)

        decontamination_by_id: dict[str, dict[str, Any]] = {
            rid: _base_decontamination_payload(
                embedding_model=self.embedding_model,
                similarity_threshold=self.similarity_threshold,
                llm_judge_model=self.llm_model,
                top_k=self.top_k,
            )
            for rid in train_record_ids
        }

        evidence_by_pair_id: dict[str, dict[str, Any]] = {}
        judge_rows: list[dict[str, Any]] = []
        bench_pair_ids: dict[str, list[str]] = {}
        candidate_count = 0
        for bench_idx, bq in enumerate(benchmarks):
            for rank in range(top_k):
                train_pos = int(neighbor_indices[bench_idx, rank])
                if train_pos < 0:
                    continue
                sim = float(distances[bench_idx, rank])
                rid = train_record_ids[train_pos]
                pair_id = f"{bq.uid}::{rid}"
                evidence = {
                    "benchmark_id": bq.uid,
                    "benchmark_source": bq.dataset_name,
                    "similarity": sim,
                    "llm_decision": None,
                }
                decontamination_by_id[rid]["candidates"].append(evidence)
                evidence_by_pair_id[pair_id] = evidence
                bench_pair_ids.setdefault(bq.uid, []).append(pair_id)
                candidate_count += 1
                judge_rows.append({
                    "pair_id": pair_id,
                    "benchmark_id": bq.uid,
                    "benchmark_source": bq.dataset_name,
                    "benchmark_question": bq.question,
                    "training_question": train_texts[train_pos],
                })

        log.info("decontam_search: candidates=%d judge_rows=%d (all top-%d judged)", candidate_count, len(judge_rows), top_k)

        judge_results = asyncio.run(
            _judge_candidates(
                judge_rows,
                api_base=self.llm_api_base,
                api_key=self.llm_api_key,
                model=self.llm_model,
                system_prompt=self.system_prompt,
                prompt_template=self.prompt_template,
                concurrency=self.llm_concurrency,
                max_retries=self.llm_max_retries,
                api_timeout=self.llm_api_timeout,
                max_tokens=self.llm_max_tokens,
                temperature=self.llm_temperature,
                top_p=self.llm_top_p,
                max_question_chars=self.max_question_chars,
                extra_body=self.llm_extra_body if self.llm_extra_body else None,
            )
        )

        contamination_hits = 0
        for pair_id, result in judge_results.items():
            evidence = evidence_by_pair_id.get(pair_id)
            if evidence is None:
                continue
            evidence["llm_decision"] = result.decision
            if result.error:
                evidence["llm_error"] = result.error
            if result.reason:
                evidence["llm_reason"] = result.reason

            benchmark_id, rid = pair_id.split("::", 1)
            payload = decontamination_by_id[rid]
            if result.decision == "yes":
                contamination_hits += 1
                cur_src = payload.get("contamination_source_id")
                cur_sim = -1.0
                if cur_src:
                    for c in payload["candidates"]:
                        if c.get("benchmark_id") == cur_src:
                            cur_sim = float(c.get("similarity", -1.0))
                            break
                if float(evidence.get("similarity", -1.0)) >= cur_sim:
                    payload["contamination_source_id"] = benchmark_id
                    payload["contamination_source_set"] = evidence.get("benchmark_source")
                    payload["status"] = "contaminated"

        benchmark_drop_stats: list[dict[str, Any]] = []
        for bq in benchmarks:
            pids = bench_pair_ids.get(bq.uid, [])
            total_judged = len(pids)
            yes_count = sum(
                1 for pid in pids
                if judge_results.get(pid) and judge_results[pid].decision == "yes"
            )
            drop_rate = yes_count / total_judged if total_judged > 0 else 0.0
            benchmark_drop_stats.append({
                "benchmark_uid": bq.uid,
                "benchmark_source": bq.dataset_name,
                "top_k": top_k,
                "judged": total_judged,
                "contaminated": yes_count,
                "drop_rate": round(drop_rate, 4),
            })

        drop_stats_path = Path(self.checkpoint_dir) / "benchmark_drop_stats.json"
        import json as _json
        drop_stats_path.write_text(_json.dumps(benchmark_drop_stats, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("decontam_search: benchmark drop stats saved to %s (%d entries)", drop_stats_path, len(benchmark_drop_stats))

        avg_drop = sum(s["drop_rate"] for s in benchmark_drop_stats) / len(benchmark_drop_stats) if benchmark_drop_stats else 0.0
        non_zero = sum(1 for s in benchmark_drop_stats if s["contaminated"] > 0)
        log.info(
            "decontam_search: avg_drop_rate=%.4f benchmarks_with_hits=%d/%d",
            avg_drop, non_zero, len(benchmark_drop_stats),
        )

        for payload in decontamination_by_id.values():
            payload["candidates"].sort(key=lambda c: float(c.get("similarity", 0.0)), reverse=True)

        updated_records = list(records)
        processor_results: list[ProcessorResult] = []
        contaminated_count = 0
        for idx in target_indices:
            rec = updated_records[idx]
            payload = decontamination_by_id[rec.record_id]
            if payload["status"] == "contaminated":
                contaminated_count += 1
                updated = rec.clone(
                    training_phase="drop",
                    filter_tag="contaminated",
                    decontamination=payload,
                )
                updated.add_trace(
                    stage="decontamination",
                    processor=self.name,
                    status="routed",
                    reason_code="contaminated",
                    details={
                        "benchmark_id": payload.get("contamination_source_id"),
                        "benchmark_source": payload.get("contamination_source_set"),
                    },
                )
                updated_records[idx] = updated
                processor_results.append(ProcessorResult(
                    keep=True, record=updated, stage="decontamination",
                    processor=self.name, reason_code="contaminated",
                    details={
                        "benchmark_id": payload.get("contamination_source_id"),
                        "benchmark_source": payload.get("contamination_source_set"),
                    },
                ))
            else:
                updated_records[idx] = rec.clone(decontamination=payload)

        log.info(
            "decontam_search: target=%d contaminated=%d clean=%d judge_yes=%d",
            len(target_indices), contaminated_count,
            len(target_indices) - contaminated_count, contamination_hits,
        )
        _ = gpu_resources
        return DatasetProcessorResult(
            kept_records=updated_records,
            processor_results=processor_results,
            artifacts={
                "benchmark_drop_stats_path": str(drop_stats_path),
                "avg_drop_rate": round(avg_drop, 4),
                "benchmarks_with_hits": non_zero,
            },
        )
