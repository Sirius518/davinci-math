"""
High-concurrency async rollout for math correctness verification.

Uses an OpenAI-compatible chat API (e.g. SGLang / vLLM) to generate
multiple solutions per math question, aggregating results via majority vote.

Key features:
  - Prompt driven by YAML config (same pattern as ``llm_classifier``)
  - Async high-concurrency via aiohttp + semaphore
  - Uses the ``n`` parameter to request multiple completions per API call,
    sharing prompt prefix KV-cache across all samples
  - Configurable: temperature, top_p, top_k, max_tokens, reasoning_effort
  - Results conform to SCHEMA.md ``verification`` structure

SGLang integration notes:
  - ``n`` > 1 lets SGLang decode multiple continuations from a single
    prompt encoding, which is far more efficient than separate requests.
  - Continuous batching handles the concurrency; we just send many
    concurrent HTTP requests and the server batches on GPU.
  - Server should be launched WITHOUT ``--reasoning-parser`` so that
    generation starts from the first token (no wasted reasoning prefix).

Pipeline YAML example::

    processors:
      - name: rollout_correctness
        prompt_path: configs/prompts/rollout_correctness.yaml
        api_base: http://10.0.0.1:8000/v1
        model: gpt-oss-120b
        num_samples: 8
        concurrency: 256
        max_retries: 3
        temperature: 0.7
        top_p: 0.95
        max_tokens: 4096
        reasoning_effort: medium
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp

from pathlib import Path

from pipeline.core.io import load_yaml, write_canonical_records
from pipeline.core.registry import register_processor
from pipeline.core.schema import (
    CanonicalRecord,
    DatasetProcessor,
    DatasetProcessorResult,
    RolloutResult,
)
from pipeline.utils.text import majority_vote, normalize_answer_text

log = logging.getLogger(__name__)

ANSWER_PATTERN = re.compile(
    r"(?im)^(?:final answer|answer)\s*[:：]\s*(.+)$",
)
_TRUNCATED_MARKER = "\n[TRUNCATED]"


# ---------------------------------------------------------------------------
# Public helpers (also exported from the package)
# ---------------------------------------------------------------------------

def parse_answer(text: str, pattern: re.Pattern | None = None) -> str:
    """Extract the final answer from model output.

    Takes the **last** match when the pattern appears multiple times
    (e.g. intermediate ``\\boxed{}`` in reasoning followed by the final one).
    """
    pat = pattern or ANSWER_PATTERN
    matches = list(pat.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    return text.splitlines()[-1].strip() if text.strip() else ""


def verify_equivalent(left: str, right: str) -> bool:
    return normalize_answer_text(left) == normalize_answer_text(right)


def build_rollout_prompt(prompt_cfg: dict[str, Any]) -> tuple[str, str]:
    """Build ``(prompt_template, system_prompt)`` from YAML config.

    If the YAML has a ``template`` key (preferred), return it directly —
    ``{problem}`` will be replaced at call-time.
    Otherwise, assemble from ``instruction / examples / suffix``.
    """
    system_prompt = prompt_cfg.get("system_prompt", "").strip()
    template = prompt_cfg.get("template", "").strip()
    if template:
        return template, system_prompt

    parts: list[str] = []
    instruction = prompt_cfg.get("instruction", "").strip()
    if instruction:
        parts.append(instruction)

    examples = prompt_cfg.get("examples", [])
    if examples:
        parts.append("\n## Examples\n")
        for ex in examples:
            q = ex.get("question", "")
            a = ex.get("answer", "")
            parts.append(f"Problem:\n{q}\nFinal Answer: {a}")

    suffix = prompt_cfg.get("suffix", "\nProblem:\n")
    parts.append(suffix)

    template = "\n".join(parts) + "{problem}"
    return template, system_prompt


# ---------------------------------------------------------------------------
# Internal async helpers
# ---------------------------------------------------------------------------

def _truncate_question(question: str, max_content_chars: int) -> str:
    budget = max(64, max_content_chars)
    if len(question) <= budget:
        return question
    return question[:budget] + _TRUNCATED_MARKER


async def _call_rollout_api(
    session: Any,
    url: str,
    payload: dict[str, Any],
    record_id: str,
    semaphore: Any,
    max_retries: int = 3,
    num_samples: int = 8,
    api_timeout: float = 600.0,
) -> tuple[str, list[dict[str, Any]], int]:
    """Send one rollout request with ``n=num_samples`` using SSE streaming.

    Returns ``(record_id, sample_dicts, attempts)``.
    Each sample_dict: ``{content, reasoning_content, finish_reason, success}``.
    """
    import json as _json

    stream_mode = payload.get("stream", False)

    for attempt in range(1, max_retries + 1):
        should_retry = False
        backoff = 0.0
        data: dict[str, Any] | None = None

        async with semaphore:
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=api_timeout),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.warning(
                            "rid=%s attempt=%d status=%d body=%s",
                            record_id, attempt, resp.status, body,
                        )
                        # 400 = client error (e.g. context length exceeded,
                        # malformed payload). Retrying won't help — give up
                        # immediately so the caller can mark the record as failed.
                        if 400 <= resp.status < 500 and resp.status != 429:
                            return record_id, [], attempt
                        should_retry = True
                        backoff = min(2 ** attempt, 30)
                    elif not stream_mode:
                        data = await resp.json()
                    else:
                        accum_content: dict[int, list[str]] = {}
                        accum_reasoning: dict[int, list[str]] = {}
                        finish_reasons: dict[int, str] = {}

                        buffer = ""
                        async for raw_chunk in resp.content:
                            buffer += raw_chunk.decode("utf-8", errors="replace")
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if not line or not line.startswith("data:"):
                                    continue
                                payload_str = line[len("data:"):].strip()
                                if payload_str == "[DONE]":
                                    break
                                try:
                                    chunk_data = _json.loads(payload_str)
                                except _json.JSONDecodeError:
                                    continue
                                for choice in chunk_data.get("choices", []):
                                    idx = choice.get("index", 0)
                                    delta = choice.get("delta", {})
                                    if delta.get("content"):
                                        accum_content.setdefault(idx, []).append(delta["content"])
                                    if delta.get("reasoning_content"):
                                        accum_reasoning.setdefault(idx, []).append(delta["reasoning_content"])
                                    fr = choice.get("finish_reason")
                                    if fr:
                                        finish_reasons[idx] = fr

                        data = {
                            "choices": [
                                {
                                    "index": idx,
                                    "message": {
                                        "content": "".join(accum_content.get(idx, [])),
                                        "reasoning_content": "".join(accum_reasoning.get(idx, [])),
                                    },
                                    "finish_reason": finish_reasons.get(idx, "unknown"),
                                }
                                for idx in range(num_samples)
                            ]
                        }
            except Exception as exc:
                log.warning("rid=%s attempt=%d error=%s", record_id, attempt, exc)
                should_retry = True
                backoff = min(2 ** attempt, 30)

        # Sleep OUTSIDE the semaphore so retries don't hold a concurrency slot
        if should_retry:
            await asyncio.sleep(backoff)
            continue

        choices = data.get("choices", []) if data else []
        if not choices:
            log.warning("rid=%s attempt=%d empty choices", record_id, attempt)
            await asyncio.sleep(min(2 ** attempt, 30))
            continue

        samples: list[dict[str, Any]] = []
        for choice in choices:
            try:
                msg = choice["message"]
                content = msg.get("content") or ""
                reasoning_content = msg.get("reasoning_content") or ""
                finish_reason = choice.get("finish_reason", "unknown")
            except (KeyError, IndexError):
                content = ""
                reasoning_content = ""
                finish_reason = "unknown"
            samples.append({
                "content": content,
                "reasoning_content": reasoning_content,
                "finish_reason": finish_reason,
                "success": bool(content or reasoning_content),
            })
        return record_id, samples, attempt

    return record_id, [], max_retries


async def _rollout_batch_streaming(
    rows: list[dict[str, Any]],
    *,
    api_base: str,
    model: str,
    prompt_template: str,
    system_prompt: str,
    num_samples: int,
    concurrency: int,
    max_retries: int,
    max_content_chars: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    reasoning_effort: str,
    separate_reasoning: bool,
    api_timeout: float,
    extra_body: dict[str, Any],
    api_key: str = "",
    split_n: bool = False,
) -> AsyncGenerator[tuple[str, list[dict[str, Any]], int], None]:
    """Yield ``(record_id, samples, attempts)`` as each record completes.

    Each row may carry its own ``samples_needed`` (e.g. partial-fill resume:
    a record that already has 5 samples only needs 3 more).  Falls back to
    *num_samples* when the row doesn't specify.

    When *split_n* is True and a row needs more than 1 sample, each missing
    slot becomes its own ``n=1`` request so the router can distribute them
    across different workers.  Results are aggregated by record_id and
    yielded once all samples arrive (or at the end for partial failures).
    """
    url = api_base.rstrip("/") + "/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Random shuffle the rows. We previously sorted by question length (LPT)
    # to reduce long-tail makespan, but the assumption "long prompt -> long
    # decode" breaks down on noisy data where very long prompts are usually
    # malformed records that immediately fail with 4xx context-length errors.
    # Concentrating those failures at the start of the stream causes a burst
    # of failures that can trip the sglang_router circuit breaker, sending
    # all subsequent requests into a 503 storm. Shuffling spreads the bad
    # records across the stream so failures are diluted in time.
    rows = list(rows)
    random.shuffle(rows)

    # Per-record expected sample count (supports partial fill)
    expected_per_record: dict[str, int] = {
        row["record_id"]: int(row.get("samples_needed", num_samples))
        for row in rows
    }

    # Aggregation state for split_n mode
    pending_samples: dict[str, list[dict[str, Any]]] = {}
    pending_attempts: dict[str, int] = {}
    yielded: set[str] = set()

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        tasks = []
        # Track which record_ids actually go through the split path so we
        # can correctly strip the "::N" suffix during aggregation.
        split_rids: set[str] = set()
        for row in rows:
            question = row["question"]
            row_n_needed = int(row.get("samples_needed", num_samples))
            if row_n_needed <= 0:
                continue
            row_use_split = split_n and row_n_needed > 1
            was_truncated = _truncate_question(question, max_content_chars)
            if len(was_truncated) < len(question):
                log.warning(
                    "rid=%s question truncated %d->%d budget=%d",
                    row["record_id"], len(question), len(was_truncated), max_content_chars,
                )
                question = was_truncated

            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": prompt_template.replace("{problem}", question),
            })

            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "n": row_n_needed,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if separate_reasoning:
                payload["separate_reasoning"] = True
            if reasoning_effort:
                payload["reasoning_effort"] = reasoning_effort
            if extra_body:
                payload.update(extra_body)

            if row_use_split:
                split_rids.add(row["record_id"])
                for si in range(row_n_needed):
                    p = dict(payload)
                    p["n"] = 1
                    # Use the same messages list (no need to deep-copy, it's read-only)
                    task_id = f"{row['record_id']}::{si}"
                    tasks.append(
                        _call_rollout_api(
                            session, url, p, task_id,
                            semaphore, max_retries,
                            num_samples=1,
                            api_timeout=api_timeout,
                        )
                    )
            else:
                tasks.append(
                    _call_rollout_api(
                        session, url, payload, row["record_id"],
                        semaphore, max_retries, row_n_needed, api_timeout,
                    )
                )

        done = 0
        total = len(tasks)
        records_total = len(rows)
        records_done = 0
        t0 = time.monotonic()

        for coro in asyncio.as_completed(tasks):
            raw_rid, samples, att = await coro
            done += 1

            if "::" in raw_rid:
                real_rid = raw_rid.rsplit("::", 1)[0]
            else:
                real_rid = raw_rid

            if real_rid in split_rids:
                pending_samples.setdefault(real_rid, []).extend(samples)
                pending_attempts[real_rid] = max(
                    pending_attempts.get(real_rid, 0), att
                )
                if len(pending_samples[real_rid]) >= expected_per_record.get(real_rid, num_samples):
                    yielded.add(real_rid)
                    records_done += 1
                    yield (
                        real_rid,
                        pending_samples.pop(real_rid),
                        pending_attempts.pop(real_rid),
                    )
            else:
                records_done += 1
                yield (real_rid, samples, att)

            if done % max(1, total // 20) == 0 or done == total:
                elapsed = time.monotonic() - t0
                speed = done / max(elapsed, 0.001)
                remaining = total - done
                eta = remaining / max(speed, 0.001)
                log.info(
                    "rollout progress %d/%d tasks (%d/%d records) "
                    "| %.1f tasks/s | ETA %.0fs",
                    done, total, records_done, records_total, speed, eta,
                )

    # Yield partially completed records (some samples may have failed)
    for rid in list(pending_samples.keys()):
        if rid not in yielded:
            yield (
                rid,
                pending_samples.pop(rid),
                pending_attempts.pop(rid, max_retries),
            )


async def _rollout_batch(
    rows: list[dict[str, Any]],
    *,
    api_base: str,
    model: str,
    prompt_template: str,
    system_prompt: str,
    num_samples: int,
    concurrency: int,
    max_retries: int,
    max_content_chars: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    reasoning_effort: str,
    separate_reasoning: bool,
    api_timeout: float,
    extra_body: dict[str, Any],
    api_key: str = "",
    split_n: bool = False,
) -> dict[str, tuple[list[dict[str, Any]], int]]:
    """Run rollout for a batch. Returns ``{record_id: (samples, attempts)}``."""
    results: dict[str, tuple[list[dict[str, Any]], int]] = {}
    async for rid, samples, att in _rollout_batch_streaming(
        rows,
        api_base=api_base,
        model=model,
        prompt_template=prompt_template,
        system_prompt=system_prompt,
        num_samples=num_samples,
        concurrency=concurrency,
        max_retries=max_retries,
        max_content_chars=max_content_chars,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        separate_reasoning=separate_reasoning,
        api_timeout=api_timeout,
        extra_body=extra_body,
        api_key=api_key,
        split_n=split_n,
    ):
        results[rid] = (samples, att)
    return results


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

@register_processor("rollout_correctness")
class RolloutCorrectnessProcessor(DatasetProcessor):
    name = "rollout_correctness"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.api_base: str = self.config.get("api_base", "http://localhost:8000/v1")
        self.model: str = self.config.get("model", "gpt-oss-120b")
        self.num_samples: int = int(self.config.get("num_samples", 8))
        self.concurrency: int = int(self.config.get("concurrency", 256))
        self.max_retries: int = int(self.config.get("max_retries", 3))
        self.temperature: float = float(self.config.get("temperature", 0.7))
        self.top_p: float = float(self.config.get("top_p", 0.95))
        self.top_k: int = int(self.config.get("top_k", -1))
        self.max_tokens: int = int(self.config.get("max_tokens", 4096))
        self.max_content_chars: int = int(self.config.get("max_content_chars", 50000))
        self.max_question_chars: int = int(self.config.get("max_question_chars", 6000))
        self.reasoning_effort: str = str(self.config.get("reasoning_effort", ""))
        self.api_timeout: float = float(self.config.get("api_timeout", 600.0))
        self.separate_reasoning: bool = bool(self.config.get("separate_reasoning", False))
        self.target_phase: str = self.config.get("target_phase", "")
        self.extra_body: dict[str, Any] = dict(self.config.get("extra_body", {}))
        self.save_every: int = int(self.config.get("save_every", 0))
        self.split_n: bool = bool(self.config.get("split_n", True))
        self.output_path: str = self.config.get("output_path", "")
        self.api_key: str = self.config.get(
            "api_key", os.environ.get("INF_API_KEY", "")
        )

        prompt_path = self.config.get(
            "prompt_path", "configs/prompts/rollout_correctness.yaml"
        )
        prompt_cfg = load_yaml(prompt_path)
        self.prompt_template, self.system_prompt = build_rollout_prompt(prompt_cfg)
        self.prompt_name = prompt_cfg.get("name", "rollout_correctness")

        answer_pattern_str = prompt_cfg.get("answer_pattern", "")
        if answer_pattern_str:
            self.answer_pattern = re.compile(
                answer_pattern_str, re.IGNORECASE | re.MULTILINE
            )
        else:
            self.answer_pattern = ANSWER_PATTERN

        log.info(
            "RolloutCorrectness: api=%s model=%s n=%d concurrency=%d "
            "temp=%.2f top_p=%.2f top_k=%d max_tokens=%d "
            "reasoning_effort=%r separate_reasoning=%s target_phase=%r "
            "split_n=%s max_question_chars=%d template_len=%d",
            self.api_base, self.model, self.num_samples, self.concurrency,
            self.temperature, self.top_p, self.top_k, self.max_tokens,
            self.reasoning_effort, self.separate_reasoning, self.target_phase,
            self.split_n, self.max_question_chars, len(self.prompt_template),
        )

    def _parse_answer(self, text: str) -> str:
        return parse_answer(text, self.answer_pattern)

    def _build_verification(
        self,
        rec: CanonicalRecord,
        samples_data: list[dict[str, Any]],
        existing_samples: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        """Parse samples, compute majority vote, build verification dict.

        If *existing_samples* (already in processed payload form) is provided,
        the new *samples_data* is appended to them — used by the partial-fill
        resume path so that previously rolled-out samples are preserved and
        only the missing slots are populated by the new API responses.

        Returns (verification_dict, raw_samples, parsed_answers).
        """
        parsed_answers: list[str] = []
        sample_payloads: list[dict[str, Any]] = []
        raw_samples: list[str] = []

        # Carry over existing samples (already in the canonical payload form)
        if existing_samples:
            for s in existing_samples:
                parsed_answers.append(s.get("answer", "") or "")
                raw_samples.append(s.get("solution", "") or "")
                sample_payloads.append(s)

        offset = len(sample_payloads)
        for i, sample in enumerate(samples_data):
            content = sample.get("content", "")
            reasoning_content = sample.get("reasoning_content", "")
            finish_reason = sample.get("finish_reason", "unknown")
            success = sample.get("success", False)

            parsed = self._parse_answer(content)
            if not parsed and reasoning_content:
                parsed = self._parse_answer(reasoning_content)

            stop_reason = "natural"
            if finish_reason in ("length", "truncated"):
                stop_reason = "truncated"

            matches_gt = verify_equivalent(parsed, rec.raw_dataset_answer)
            raw_samples.append(content)
            parsed_answers.append(parsed)
            sample_payloads.append({
                "index": offset + i,
                "answer": parsed,
                "solution": content,
                "reasoning": reasoning_content if reasoning_content else content,
                "success": success,
                "stop_reason": stop_reason,
                "judge_correct": True if matches_gt else None,
            })

        majority_answer, majority_count = majority_vote(parsed_answers)
        is_correct = verify_equivalent(majority_answer, rec.raw_dataset_answer)

        correct_count = sum(
            1 for pa in parsed_answers
            if verify_equivalent(pa, rec.raw_dataset_answer)
        )
        actual_n = max(len(sample_payloads), 1)
        pass_ratio = float(correct_count) / float(actual_n)

        verification = dict(rec.verification)
        verification.update({
            "model": self.model,
            "num_samples": actual_n,
            "raw_dataset_answer": rec.raw_dataset_answer,
            "majority_answer": majority_answer,
            "majority_count": majority_count,
            "majority_matches_gt": is_correct,
            "pass_ratio": pass_ratio,
            "fallback_model": None,
            "used_fallback": False,
            "samples": sample_payloads,
        })
        return verification, raw_samples, parsed_answers

    def _apply_api_results(
        self,
        records: list[CanonicalRecord],
        updated_records: list[CanonicalRecord],
        row_idx: dict[str, int],
        api_results: dict[str, tuple[list[dict[str, Any]], int]],
    ) -> tuple[list[RolloutResult], int]:
        """Apply API results to records in-place, return rollout results and error count."""
        rollout_results: list[RolloutResult] = []
        error_count = 0

        for rid, (samples_data, attempts) in api_results.items():
            idx = row_idx[rid]
            rec = records[idx]

            if not samples_data:
                rec.add_trace(
                    stage="evaluate",
                    processor=self.name,
                    status="api_error",
                    details={"attempts": attempts},
                )
                error_count += 1
                continue

            existing_samples = (rec.verification or {}).get("samples") or []
            verification, raw_samples, parsed_answers = self._build_verification(
                rec, samples_data, existing_samples=existing_samples
            )
            is_correct = verification["majority_matches_gt"]
            majority_answer = verification["majority_answer"]

            if not is_correct:
                rec.add_trace(
                    stage="evaluate",
                    processor=self.name,
                    status="flagged",
                    reason_code="gt_suspect",
                    details={
                        "majority_answer": majority_answer,
                        "majority_count": verification["majority_count"],
                        "pass_ratio": verification["pass_ratio"],
                    },
                )
            else:
                rec.add_trace(
                    stage="evaluate",
                    processor=self.name,
                    status="kept",
                    details={
                        "majority_answer": majority_answer,
                        "majority_count": verification["majority_count"],
                        "pass_ratio": verification["pass_ratio"],
                    },
                )

            updated = rec.clone(verification=verification)
            updated_records[idx] = updated

            rollout_results.append(
                RolloutResult(
                    record_id=rid,
                    backend="openai_compat",
                    model=self.model,
                    prompt_name=self.prompt_name,
                    samples=raw_samples,
                    parsed_answers=parsed_answers,
                    majority_answer=majority_answer,
                    majority_count=verification["majority_count"],
                    is_correct=is_correct,
                    metadata={"raw_dataset_answer": rec.raw_dataset_answer},
                )
            )

        return rollout_results, error_count

    def _save_snapshot(self, records: list[CanonicalRecord], tag: str) -> None:
        """Save an intermediate snapshot of all records to disk."""
        if not self.output_path:
            return
        out = Path(self.output_path)
        snapshot_path = out.with_name(f"{out.stem}_snapshot{out.suffix}")
        try:
            write_canonical_records(records, snapshot_path)
            log.info("Snapshot saved: %s (%d records, %s)", snapshot_path, len(records), tag)
        except Exception:
            log.exception("Failed to save snapshot %s", snapshot_path)

    def process(
        self,
        records: list[CanonicalRecord],
        *,
        pipeline_artifacts: dict[str, Any] | None = None,
    ) -> DatasetProcessorResult:
        rows: list[dict[str, Any]] = []
        row_idx: dict[str, int] = {}
        skipped = 0
        partial_filled = 0
        skipped_too_long = 0
        for i, rec in enumerate(records):
            if self.target_phase and rec.training_phase != self.target_phase:
                continue
            existing_v = rec.verification or {}
            existing_samples = existing_v.get("samples") or []
            n_existing = len(existing_samples)
            if n_existing >= self.num_samples:
                skipped += 1
                continue
            # Hard drop overlong questions before they reach the API.
            # Avoids 400 error storms that can trip the router circuit breaker.
            n_chars = len(rec.question)
            if n_chars > self.max_question_chars:
                rec.add_trace(
                    stage="evaluate",
                    processor=self.name,
                    status="skipped",
                    reason_code="question_too_long",
                    details={
                        "chars": n_chars,
                        "limit": self.max_question_chars,
                    },
                )
                skipped_too_long += 1
                continue
            n_needed = self.num_samples - n_existing
            if n_existing > 0:
                partial_filled += 1
            rows.append({
                "record_id": rec.record_id,
                "question": rec.question,
                "samples_needed": n_needed,
            })
            row_idx[rec.record_id] = i

        log.info(
            "RolloutCorrectness: total=%d target_phase=%r matched=%d "
            "skipped_done=%d partial_fill=%d skipped_too_long=%d",
            len(records), self.target_phase, len(rows), skipped,
            partial_filled, skipped_too_long,
        )

        if not rows:
            return DatasetProcessorResult(kept_records=list(records))

        updated_records: list[CanonicalRecord] = list(records)
        all_rollout_results: list[RolloutResult] = []
        total_errors = 0

        save_every = self.save_every if self.save_every > 0 else len(rows)

        log.info(
            "RolloutCorrectness: %d rows, continuous mode (split_n=%s), "
            "snapshot every %d records",
            len(rows), self.split_n, save_every,
        )

        async def _process_streaming() -> None:
            nonlocal total_errors
            records_done = 0
            since_last_save = 0

            async for rid, samples_data, attempts in _rollout_batch_streaming(
                rows,
                api_base=self.api_base,
                model=self.model,
                prompt_template=self.prompt_template,
                system_prompt=self.system_prompt,
                num_samples=self.num_samples,
                concurrency=self.concurrency,
                max_retries=self.max_retries,
                max_content_chars=self.max_content_chars,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
                separate_reasoning=self.separate_reasoning,
                api_timeout=self.api_timeout,
                extra_body=self.extra_body,
                api_key=self.api_key,
                split_n=self.split_n,
            ):
                single_result = {rid: (samples_data, attempts)}
                chunk_results, chunk_errors = self._apply_api_results(
                    records, updated_records, row_idx, single_result,
                )
                all_rollout_results.extend(chunk_results)
                total_errors += chunk_errors

                records_done += 1
                since_last_save += 1

                if since_last_save >= save_every and records_done < len(rows):
                    since_last_save = 0
                    correct = sum(1 for r in all_rollout_results if r.is_correct)
                    flagged = len(all_rollout_results) - correct
                    log.info(
                        "Snapshot trigger: %d/%d done, correct=%d flagged=%d "
                        "errors=%d",
                        records_done, len(rows), correct, flagged, total_errors,
                    )
                    snapshot_copy = list(updated_records)
                    await asyncio.to_thread(
                        self._save_snapshot,
                        snapshot_copy,
                        f"continuous {records_done}/{len(rows)}",
                    )

        asyncio.run(_process_streaming())

        correct_count = sum(1 for r in all_rollout_results if r.is_correct)
        flagged_count = len(all_rollout_results) - correct_count
        log.info(
            "RolloutCorrectness done: total=%d correct=%d flagged=%d errors=%d",
            len(rows), correct_count, flagged_count, total_errors,
        )

        return DatasetProcessorResult(
            kept_records=updated_records,
            artifacts={"rollout_results": all_rollout_results},
        )
