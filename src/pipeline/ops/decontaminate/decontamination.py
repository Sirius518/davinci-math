from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from pipeline.core.io import load_yaml
from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, DatasetProcessor, DatasetProcessorResult, ProcessorResult
from pipeline.ops.decontaminate.embedding import embed_all, l2_normalize

log = logging.getLogger(__name__)

DEFAULT_BENCHMARK_DATASETS = [
    "aime2024",
    "aime2025",
    "aime2026",
    "amc23",
    "beyondaime",
    "hmmtfeb2025",
    "hmmtfeb2026",
    "hmmtnov2025",
    "imo2025",
    "imoanswerbench",
    "math500",
    "minerva",
    "olympiadbench",
]


@dataclass(slots=True)
class BenchmarkQuestion:
    uid: str
    dataset_name: str
    question: str
    ground_truth: str


@dataclass(slots=True)
class JudgeResult:
    decision: str
    reason: str
    raw_response: str
    attempts: int
    error: str = ""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_benchmark_paths() -> list[str]:
    base_dir = _project_root() / "data" / "data_same_format"
    return [str(base_dir / f"{name}.parquet") for name in DEFAULT_BENCHMARK_DATASETS]


def _extract_prompt_parts(prompt_cfg: dict[str, Any]) -> tuple[str, str]:
    system_prompt = str(prompt_cfg.get("system_prompt", "")).strip()
    template = str(prompt_cfg.get("template", "")).strip()
    output_format = str(prompt_cfg.get("output_format", "")).strip()
    if output_format:
        escaped_fmt = output_format.replace("{", "{{").replace("}", "}}")
        template = f"{template}\n\n{escaped_fmt}".strip()
    return system_prompt, template


def _build_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["yes", "no"]},
            "reason": {"type": "string"},
        },
        "required": ["decision", "reason"],
    }


def _parse_judge_response(text: str) -> tuple[str | None, str]:
    try:
        payload = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, ""
    if not isinstance(payload, dict):
        return None, ""
    decision = str(payload.get("decision", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if decision not in {"yes", "no"}:
        return None, reason
    return decision, reason


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n[TRUNCATED]"
    keep = max(0, max_chars - len(marker))
    return f"{text[:keep].rstrip()}{marker}"


def _read_arrow_table(path: str, *, columns: list[str]) -> Any:
    target = Path(path)
    if target.is_dir():
        dataset = ds.dataset(target, format="parquet")
        return dataset.to_table(columns=columns, use_threads=True)
    return pq.read_table(target, columns=columns)


def load_benchmark_questions(paths: list[str]) -> list[BenchmarkQuestion]:
    benchmarks: list[BenchmarkQuestion] = []
    for path in paths:
        table = _read_arrow_table(path, columns=["question", "uid", "ground_truth", "dataset_name"])
        questions = table.column("question").to_pylist()
        uids = table.column("uid").to_pylist()
        answers = table.column("ground_truth").to_pylist()
        datasets = table.column("dataset_name").to_pylist()
        for uid, dataset_name, question, ground_truth in zip(uids, datasets, questions, answers, strict=False):
            question_text = str(question or "").strip()
            if not question_text:
                continue
            benchmarks.append(
                BenchmarkQuestion(
                    uid=str(uid or ""),
                    dataset_name=str(dataset_name or ""),
                    question=question_text,
                    ground_truth=str(ground_truth or ""),
                )
            )
    return benchmarks


def _build_faiss_index(
    embeddings: np.ndarray,
    *,
    gpu_id: int,
) -> tuple[Any, Any | None]:
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError("faiss is required for the decontamination processor") from exc

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("expected non-empty 2D embeddings array")
    dim = int(embeddings.shape[1])
    cpu_index = faiss.IndexFlatIP(dim)
    resources = None
    if gpu_id >= 0 and hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu"):
        try:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, gpu_id, cpu_index)
            index.add(embeddings)
            log.info("decontamination using FAISS GPU index on device %d", gpu_id)
            return index, resources
        except Exception as exc:
            log.warning("failed to initialize FAISS GPU index, falling back to CPU: %s", exc)
    cpu_index.add(embeddings)
    log.info("decontamination using FAISS CPU index")
    return cpu_index, resources


async def _call_judge_api(
    session: Any,
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    pair_id: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    api_timeout: int,
) -> tuple[str, JudgeResult]:
    import aiohttp

    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=api_timeout),
                ) as response:
                    if response.status != 200:
                        body = await response.text()
                        log.warning(
                            "pair=%s attempt=%d status=%d body=%s",
                            pair_id[:24],
                            attempt,
                            response.status,
                            body[:400],
                        )
                        continue
                    data = await response.json()
            except Exception as exc:
                log.warning("pair=%s attempt=%d error=%s", pair_id[:24], attempt, exc)
                continue

        try:
            finish_reason = str(data["choices"][0].get("finish_reason", ""))
            raw = str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            log.warning("pair=%s attempt=%d bad response structure", pair_id[:24], attempt)
            continue
        if finish_reason == "length":
            log.warning("pair=%s attempt=%d truncated", pair_id[:24], attempt)
            continue
        decision, reason = _parse_judge_response(raw)
        if decision is not None:
            return pair_id, JudgeResult(
                decision=decision,
                reason=reason,
                raw_response=raw,
                attempts=attempt,
            )
        log.warning("pair=%s attempt=%d invalid response=%s", pair_id[:24], attempt, raw[:200])

    return pair_id, JudgeResult(
        decision="no",
        reason="api_error",
        raw_response="",
        attempts=max_retries,
        error="api_error",
    )


async def _judge_candidates(
    rows: list[dict[str, Any]],
    *,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    prompt_template: str,
    concurrency: int,
    max_retries: int,
    api_timeout: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_question_chars: int,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, JudgeResult]:
    import aiohttp

    if not rows:
        return {}

    url = f"{api_base.rstrip('/')}/chat/completions"
    semaphore = asyncio.Semaphore(max(1, concurrency))
    connector = aiohttp.TCPConnector(limit=max(8, concurrency + 32), ttl_dns_cache=300)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "decontamination_judge",
            "schema": _build_json_schema(),
        },
    }

    results: dict[str, JudgeResult] = {}
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for row in rows:
            user_prompt = prompt_template.format(
                benchmark_source=row["benchmark_source"],
                benchmark_id=row["benchmark_id"],
                benchmark_question=_truncate_text(row["benchmark_question"], max_question_chars),
                training_question=_truncate_text(row["training_question"], max_question_chars),
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
            if extra_body:
                payload.update(extra_body)
            tasks.append(
                _call_judge_api(
                    session,
                    url=url,
                    payload=payload,
                    headers=headers,
                    pair_id=row["pair_id"],
                    semaphore=semaphore,
                    max_retries=max_retries,
                    api_timeout=api_timeout,
                )
            )
        total = len(tasks)
        started_at = time.monotonic()
        done = 0
        for task in asyncio.as_completed(tasks):
            pair_id, result = await task
            results[pair_id] = result
            done += 1
            if done == total or done % max(1, min(200, total)) == 0:
                elapsed = time.monotonic() - started_at
                speed = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / speed if speed > 0 else 0.0
                log.info(
                    "Decontamination judge progress %d/%d | %.1f req/s | ETA %.0fs",
                    done,
                    total,
                    speed,
                    eta,
                )
    return results


def _base_decontamination_payload(
    *,
    embedding_model: str,
    similarity_threshold: float | None,
    llm_judge_model: str,
    top_k: int,
) -> dict[str, Any]:
    return {
        "status": "clean",
        "embedding_model": embedding_model,
        "similarity_threshold": similarity_threshold,
        "llm_judge_model": llm_judge_model,
        "top_k": top_k,
        "candidates": [],
        "contamination_source_id": None,
        "contamination_source_set": None,
    }


@register_processor("decontamination")
class DecontaminationProcessor(DatasetProcessor):
    name = "decontamination"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        configured_paths = [str(path) for path in self.config.get("benchmark_paths", [])]
        self.benchmark_paths = configured_paths or _default_benchmark_paths()
        self.target_phases = {str(value) for value in self.config.get("target_phases", ["", "midtrain"])}

        self.embedding_api_base = str(self.config.get("embedding_api_base", "http://localhost:8000/v1"))
        self.embedding_api_key = str(self.config.get("embedding_api_key", ""))
        self.embedding_model = str(self.config.get("embedding_model", "text-embedding-3-large"))
        self.embedding_batch_size = int(self.config.get("embedding_batch_size", 256))
        self.embedding_concurrency = int(self.config.get("embedding_concurrency", 16))
        self.embedding_timeout = int(self.config.get("embedding_timeout", 120))
        self.embedding_max_retries = int(self.config.get("embedding_max_retries", 3))
        self.encoding_format = str(self.config.get("encoding_format", "float"))
        raw_embedding_extra_body = self.config.get("embedding_extra_body", {}) or {}
        self.embedding_extra_body = dict(raw_embedding_extra_body)

        self.top_k = int(self.config.get("top_k", 10))
        threshold_value = self.config.get("similarity_threshold")
        self.similarity_threshold = None if threshold_value in (None, "", "null") else float(threshold_value)
        self.faiss_gpu_id = int(self.config.get("faiss_gpu_id", 0))

        self.llm_api_base = str(self.config.get("llm_api_base", "http://localhost:8000/v1"))
        self.llm_api_key = str(self.config.get("llm_api_key", ""))
        self.llm_model = str(self.config.get("llm_model", "gpt-oss-20b"))
        self.llm_concurrency = int(self.config.get("llm_concurrency", 256))
        self.llm_max_retries = int(self.config.get("llm_max_retries", 3))
        self.llm_api_timeout = int(self.config.get("llm_api_timeout", 120))
        self.llm_max_tokens = int(self.config.get("llm_max_tokens", 96))
        self.llm_temperature = float(self.config.get("llm_temperature", 0.0))
        self.llm_top_p = float(self.config.get("llm_top_p", 1.0))
        self.max_question_chars = int(self.config.get("max_question_chars", 4000))

        self.checkpoint_dir = str(self.config.get("checkpoint_dir", "") or "").strip()

        prompt_path = str(
            self.config.get(
                "prompt_path",
                "configs/prompts/decontamination_judge.yaml",
            )
        )
        prompt_cfg = load_yaml(prompt_path)
        self.system_prompt, self.prompt_template = _extract_prompt_parts(prompt_cfg)

    def _resolve_benchmark_paths(self) -> list[str]:
        resolved: list[str] = []
        for path in self.benchmark_paths:
            target = Path(path)
            if target.exists():
                resolved.append(str(target))
                continue
            log.warning("benchmark path missing and will be skipped: %s", path)
        if not resolved:
            raise FileNotFoundError("no benchmark parquet files found for decontamination")
        return resolved

    def _checkpoint_path(self, name: str) -> str | None:
        if not self.checkpoint_dir:
            return None
        base = Path(self.checkpoint_dir)
        safe_model = self.embedding_model.replace("/", "_")
        return str(base / f"{name}_{safe_model}.npy")

    def process(
        self,
        records: list[CanonicalRecord],
        *,
        pipeline_artifacts: dict[str, Any] | None = None,
    ) -> DatasetProcessorResult:
        resolved_benchmark_paths = self._resolve_benchmark_paths()
        benchmarks = load_benchmark_questions(resolved_benchmark_paths)
        if not benchmarks:
            raise ValueError("no benchmark questions were loaded for decontamination")

        updated_records = list(records)
        processor_results: list[ProcessorResult] = []
        target_indices: list[int] = []
        train_texts: list[str] = []
        train_record_ids: list[str] = []
        decontamination_by_id: dict[str, dict[str, Any]] = {}

        for idx, record in enumerate(records):
            if record.training_phase not in self.target_phases:
                continue
            target_indices.append(idx)
            train_record_ids.append(record.record_id)
            train_texts.append(record.question or "")
            decontamination_by_id[record.record_id] = _base_decontamination_payload(
                embedding_model=self.embedding_model,
                similarity_threshold=self.similarity_threshold,
                llm_judge_model=self.llm_model,
                top_k=self.top_k,
            )

        log.info(
            "Decontamination: total_records=%d target_records=%d benchmarks=%d top_k=%d threshold=%s",
            len(records),
            len(target_indices),
            len(benchmarks),
            self.top_k,
            self.similarity_threshold,
        )

        if not target_indices:
            return DatasetProcessorResult(kept_records=updated_records)

        train_embeddings = asyncio.run(
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
                extra_body=self.embedding_extra_body,
                checkpoint_path=self._checkpoint_path("train"),
                progress_name="train_embeddings",
            )
        )
        benchmark_embeddings = asyncio.run(
            embed_all(
                [item.question for item in benchmarks],
                api_base=self.embedding_api_base,
                model=self.embedding_model,
                batch_size=self.embedding_batch_size,
                concurrency=self.embedding_concurrency,
                api_key=self.embedding_api_key,
                timeout_seconds=self.embedding_timeout,
                max_retries=self.embedding_max_retries,
                encoding_format=self.encoding_format,
                extra_body=self.embedding_extra_body,
                checkpoint_path=self._checkpoint_path("benchmark"),
                progress_name="benchmark_embeddings",
            )
        )

        train_embeddings = l2_normalize(train_embeddings)
        benchmark_embeddings = l2_normalize(benchmark_embeddings)

        index, gpu_resources = _build_faiss_index(train_embeddings, gpu_id=self.faiss_gpu_id)

        top_k = min(self.top_k, len(train_record_ids))
        distances, neighbor_indices = index.search(benchmark_embeddings, top_k)

        evidence_by_pair_id: dict[str, dict[str, Any]] = {}
        judge_rows: list[dict[str, Any]] = []
        candidate_count = 0
        for bench_idx, benchmark in enumerate(benchmarks):
            for rank in range(top_k):
                train_pos = int(neighbor_indices[bench_idx, rank])
                if train_pos < 0:
                    continue
                similarity = float(distances[bench_idx, rank])
                record_id = train_record_ids[train_pos]
                pair_id = f"{benchmark.uid}::{record_id}"
                evidence = {
                    "benchmark_id": benchmark.uid,
                    "benchmark_source": benchmark.dataset_name,
                    "similarity": similarity,
                    "llm_decision": None,
                }
                decontamination_by_id[record_id]["candidates"].append(evidence)
                evidence_by_pair_id[pair_id] = evidence
                candidate_count += 1
                if self.similarity_threshold is not None and similarity < self.similarity_threshold:
                    continue
                judge_rows.append(
                    {
                        "pair_id": pair_id,
                        "benchmark_id": benchmark.uid,
                        "benchmark_source": benchmark.dataset_name,
                        "benchmark_question": benchmark.question,
                        "training_question": train_texts[train_pos],
                    }
                )

        log.info(
            "Decontamination retrieval: candidates=%d judge_rows=%d",
            candidate_count,
            len(judge_rows),
        )

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
            if result.raw_response:
                evidence["llm_raw_response"] = result.raw_response

            benchmark_id, record_id = pair_id.split("::", 1)
            payload = decontamination_by_id[record_id]
            if result.decision == "yes":
                contamination_hits += 1
                current_source_id = payload.get("contamination_source_id")
                current_similarity = -1.0
                if current_source_id:
                    for candidate in payload["candidates"]:
                        if candidate.get("benchmark_id") == current_source_id:
                            current_similarity = float(candidate.get("similarity", -1.0))
                            break
                if float(evidence.get("similarity", -1.0)) >= current_similarity:
                    payload["contamination_source_id"] = benchmark_id
                    payload["contamination_source_set"] = evidence.get("benchmark_source")
                    payload["status"] = "contaminated"

        contaminated_records = 0
        for record_id, payload in decontamination_by_id.items():
            payload["candidates"].sort(
                key=lambda item: float(item.get("similarity", 0.0)),
                reverse=True,
            )

        for idx in target_indices:
            record = updated_records[idx]
            payload = decontamination_by_id[record.record_id]
            if payload["status"] == "contaminated":
                contaminated_records += 1
                updated = record.clone(
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
                processor_results.append(
                    ProcessorResult(
                        keep=True,
                        record=updated,
                        stage="decontamination",
                        processor=self.name,
                        reason_code="contaminated",
                        details={
                            "benchmark_id": payload.get("contamination_source_id"),
                            "benchmark_source": payload.get("contamination_source_set"),
                        },
                    )
                )
                continue

            updated_records[idx] = record.clone(decontamination=payload)

        log.info(
            "Decontamination finished: target=%d contaminated=%d clean=%d judge_yes=%d",
            len(target_indices),
            contaminated_records,
            len(target_indices) - contaminated_records,
            contamination_hits,
        )
        _ = gpu_resources
        return DatasetProcessorResult(
            kept_records=updated_records,
            processor_results=processor_results,
        )
