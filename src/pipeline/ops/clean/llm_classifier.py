"""
LLM-based math question classifier (DatasetProcessor).

Uses an OpenAI-compatible chat API (e.g. SGLang-served model) to
reclassify records.  Fully driven by the prompt YAML — works for both:

  1. posttrain reclassification  (posttrain_filter.yaml)
     target_phase: ""   →  labels: posttrain / midtrain / drop
  2. midtrain filtering          (midtrain_filter.yaml)
     target_phase: midtrain  →  labels: keep / drop

SGLang integration notes:
  - Uses ``response_format: json_schema`` for guaranteed structured output
    via xgrammar.  Server must be launched WITHOUT ``--reasoning-parser``
    so that json_schema constrains from the first token (no wasted
    reasoning tokens).
  - Batch inference is handled by SGLang's continuous batching — we send
    concurrent async requests which the server batches on GPU.

Pipeline YAML example::

    processors:
      - name: llm_classifier
        prompt_path: configs/prompts/posttrain_filter.yaml
        target_phase: ""          # which training_phase to process
        api_base: http://10.0.0.1:8000/v1
        model: gpt-oss-120b
        concurrency: 256
        max_retries: 3
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from pipeline.core.io import load_yaml
from pipeline.core.registry import register_processor
from pipeline.core.schema import (
    CanonicalRecord,
    DatasetProcessor,
    DatasetProcessorResult,
    ProcessorResult,
)

log = logging.getLogger(__name__)

_RESPONSE_RE = re.compile(
    r'\{\s*"label"\s*:\s*"([^"]+)"\s*,\s*"tag"\s*:\s*"([^"]+)"\s*\}',
    re.IGNORECASE,
)
_RESPONSE_RE_ALT = re.compile(
    r'\{\s*"tag"\s*:\s*"([^"]+)"\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}',
    re.IGNORECASE,
)
_TAG_BRACKET_RE = re.compile(r"\[(\w+)\]")
_TRUNCATED_MARKER = "\n[TRUNCATED]"


# ---------------------------------------------------------------------------
# Schema extraction from prompt YAML
# ---------------------------------------------------------------------------

def _extract_tags_from_definition(text: str) -> set[str]:
    """Pull [tag_name] entries from a category definition block."""
    tags = set(_TAG_BRACKET_RE.findall(text))
    if "Tag:" in text:
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("tag:"):
                tag = line.split(":", 1)[1].strip()
                if tag:
                    tags.add(tag)
    return tags


def parse_schema(prompt_cfg: dict[str, Any]) -> dict[str, frozenset[str]]:
    """Derive {label: frozenset_of_tags} from the prompt YAML."""
    cat_defs = prompt_cfg.get("category_definitions", {})
    schema: dict[str, frozenset[str]] = {}
    for label, defn in cat_defs.items():
        tags = _extract_tags_from_definition(str(defn))
        schema[label] = frozenset(tags) if tags else frozenset({label})
    return schema


def build_json_schema(schema: dict[str, frozenset[str]]) -> dict[str, Any]:
    """Build a JSON Schema dict for ``response_format`` from the parsed schema.

    Creates a flat schema with ``enum`` constraints on both *label* and *tag*.
    Label-tag consistency is still validated in code after parsing.
    """
    all_labels = sorted(schema.keys())
    all_tags = sorted({tag for tags in schema.values() for tag in tags})
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": all_labels},
            "tag": {"type": "string", "enum": all_tags},
        },
        "required": ["label", "tag"],
    }


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def build_prompt_prefix(prompt_cfg: dict[str, Any]) -> tuple[str, str]:
    """Assemble the static prefix and answer separator from the prompt YAML.

    Returns (prefix, answer_separator).
    """
    parts: list[str] = []
    parts.append(prompt_cfg.get("instruction", "").strip())

    cat_defs = prompt_cfg.get("category_definitions", {})
    parts.append("\n## Categories\n")
    for label, defn in cat_defs.items():
        parts.append(f"**{label}**\n{str(defn).strip()}\n")

    output_fmt = prompt_cfg.get("output_format", "")
    if output_fmt:
        parts.append(f"## Output Format\n\n{output_fmt.strip()}\n")

    examples = prompt_cfg.get("examples", [])
    if examples:
        parts.append("## Examples\n")
        ans_sep = prompt_cfg.get("answer_separator", "\nAnswer: ")
        for ex in examples:
            q = ex.get("question", "")
            a = ex.get("answer", "")
            lab = ex.get("label", "")
            tag = ex.get("tag", "")
            parts.append(f"Question: {q}{ans_sep}{a}")
            parts.append(f'{{"label": "{lab}", "tag": "{tag}"}}\n')

    suffix = prompt_cfg.get("suffix", "Now classify the following sample.\n\nQuestion: ")
    parts.append(suffix)

    answer_separator = prompt_cfg.get("answer_separator", "\nAnswer: ")
    return "\n".join(parts), answer_separator


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _resolve_label_for_tag(tag: str, schema: dict[str, frozenset[str]]) -> str | None:
    """Given a valid tag, find the label it belongs to."""
    for label, tags in schema.items():
        if tag in tags:
            return label
    return None


def extract_label_and_tag(
    text: str,
    schema: dict[str, frozenset[str]],
) -> tuple[str | None, str | None]:
    """Parse model response. Returns (None, None) when invalid.

    When the tag is valid but paired with a wrong label, auto-corrects
    the label based on the tag (avoids wasting retries on cross-field
    mismatches that json_schema enum constraints cannot prevent).
    """
    text = text.strip()
    label: str | None = None
    tag: str | None = None

    for regex, label_idx, tag_idx in [
        (_RESPONSE_RE, 1, 2),
        (_RESPONSE_RE_ALT, 2, 1),
    ]:
        m = regex.search(text)
        if m:
            label = m.group(label_idx).lower()
            tag = m.group(tag_idx).lower()
            break

    if label is None:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                label = str(obj.get("label", "")).lower()
                tag = str(obj.get("tag", "")).lower()
        except (json.JSONDecodeError, ValueError):
            return None, None

    if label is None or tag is None:
        return None, None

    if label in schema and tag in schema[label]:
        return label, tag

    correct_label = _resolve_label_for_tag(tag, schema)
    if correct_label is not None:
        log.debug("auto-corrected label %r→%r for tag %r", label, correct_label, tag)
        return correct_label, tag

    return None, None


def _truncate_content(
    question: str,
    answer: str,
    answer_separator: str,
    max_content_chars: int,
) -> tuple[str, str, bool]:
    """Trim question/answer to stay under the configured character budget.

    The answer is truncated first because it is usually less important for
    coarse classification than the question text itself.
    """
    if max_content_chars <= 0:
        return question, answer, False

    current_len = len(question) + len(answer_separator) + len(answer) + 1
    if current_len <= max_content_chars:
        return question, answer, False

    marker = _TRUNCATED_MARKER
    available = max_content_chars - len(answer_separator) - 1
    if available <= len(marker):
        return marker, "", True

    truncated_question = question
    truncated_answer = answer
    target_qa_len = available - len(marker)
    if target_qa_len <= 0:
        return marker, "", True

    overflow = len(truncated_question) + len(truncated_answer) - target_qa_len
    if overflow > 0 and truncated_answer:
        keep_answer_len = max(0, len(truncated_answer) - overflow)
        truncated_answer = truncated_answer[:keep_answer_len].rstrip()
        overflow -= len(answer) - len(truncated_answer)

    if overflow > 0:
        keep_question_len = max(0, len(truncated_question) - overflow)
        truncated_question = truncated_question[:keep_question_len].rstrip()

    if truncated_answer:
        truncated_answer = f"{truncated_answer}{marker}"
    else:
        truncated_question = f"{truncated_question}{marker}".strip() or marker

    return truncated_question, truncated_answer, True


# ---------------------------------------------------------------------------
# Async engine
# ---------------------------------------------------------------------------

async def _call_api(
    session,
    url: str,
    payload: dict,
    record_id: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    schema: dict[str, frozenset[str]],
) -> tuple[str, str, str, str, int]:
    """Returns (record_id, label, tag, raw_response, attempts)."""
    import aiohttp

    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.warning(
                            "rid=%s attempt=%d status=%d body=%s",
                            record_id[:12],
                            attempt,
                            resp.status,
                            body[:400],
                        )
                        continue
                    data = await resp.json()
            except Exception as exc:
                log.warning("rid=%s attempt=%d error=%s", record_id[:12], attempt, exc)
                continue

        try:
            finish_reason = data["choices"][0].get("finish_reason", "")
            raw = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            log.warning("rid=%s attempt=%d bad response structure", record_id[:12], attempt)
            continue

        if finish_reason == "length":
            log.warning("rid=%s attempt=%d truncated (finish_reason=length)", record_id[:12], attempt)
            continue

        label, tag = extract_label_and_tag(raw, schema)
        if label is not None and tag is not None:
            return record_id, label, tag, raw, attempt
        log.warning("rid=%s attempt=%d invalid response: %s", record_id[:12], attempt, raw[:120])

    return record_id, "api_error", "api_error", "", max_retries


async def _classify_batch(
    rows: list[dict[str, str]],
    api_base: str,
    model: str,
    prompt_prefix: str,
    answer_separator: str,
    concurrency: int,
    max_retries: int,
    max_content_chars: int,
    schema: dict[str, frozenset[str]],
    response_format: dict[str, Any] | None = None,
) -> dict[str, tuple[str, str, str, int]]:
    """Classify a batch. Returns {record_id: (label, tag, raw, attempts)}.

    SGLang handles continuous batching on the GPU side — sending many
    concurrent requests IS batch inference.
    """
    import aiohttp

    url = f"{api_base}/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 32, ttl_dns_cache=300)

    results: dict[str, tuple[str, str, str, int]] = {}

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for row in rows:
            question, answer, was_truncated = _truncate_content(
                question=row["question"],
                answer=row["answer"],
                answer_separator=answer_separator,
                max_content_chars=max_content_chars,
            )
            if was_truncated:
                log.warning(
                    "rid=%s content truncated q_chars=%d->%d a_chars=%d->%d budget=%d",
                    row["record_id"][:12],
                    len(row["question"]),
                    len(question),
                    len(row["answer"]),
                    len(answer),
                    max_content_chars,
                )
            full_prompt = (
                prompt_prefix
                + question
                + answer_separator
                + answer
                + "\n"
            )
            payload: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 48,
            }
            if response_format is not None:
                payload["response_format"] = response_format
            tasks.append(
                _call_api(session, url, payload, row["record_id"],
                          semaphore, max_retries, schema)
            )

        done = 0
        total = len(tasks)
        t0 = time.monotonic()
        for coro in asyncio.as_completed(tasks):
            rid, label, tag, raw, att = await coro
            results[rid] = (label, tag, raw, att)
            done += 1
            if done % 2000 == 0 or done == total:
                elapsed = time.monotonic() - t0
                speed = done / elapsed if elapsed > 0 else 0
                remaining = total - done
                eta = remaining / speed if speed > 0 else 0
                log.info(
                    "progress %d/%d | done %d | remaining %d | %.1f req/s | ETA %.0fs",
                    done, total, done, remaining, speed, eta,
                )

    return results


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

@register_processor("llm_classifier")
class LLMClassifierProcessor(DatasetProcessor):
    """Generic LLM classifier driven entirely by the prompt YAML.

    Config keys:
        prompt_path    – path to the prompt YAML
        target_phase   – which training_phase to process ("" for posttrain,
                         "midtrain" for midtrain filter)
        label_actions  – mapping from model label → training_phase to set.
                         Special value "keep" means leave training_phase as-is.
                         Defaults are inferred from label names.
        use_structured_output – if True (default), use ``response_format:
                         json_schema`` for guaranteed structured output.
                         Server must be launched WITHOUT ``--reasoning-parser``.
        api_base, model, concurrency, max_retries – API settings
        max_content_chars – max chars kept from question+answer per request
    """
    name = "llm_classifier"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.api_base: str = self.config.get("api_base", "http://localhost:8000/v1")
        self.model: str = self.config.get("model", "gpt-oss-120b")
        self.concurrency: int = int(self.config.get("concurrency", 256))
        self.max_retries: int = int(self.config.get("max_retries", 3))
        self.max_content_chars: int = int(self.config.get("max_content_chars", 8000))
        self.target_phase: str = self.config.get("target_phase", "")
        self.use_structured_output: bool = self.config.get("use_structured_output", True)

        prompt_path = self.config.get("prompt_path", "configs/prompts/posttrain_filter.yaml")
        prompt_cfg = load_yaml(prompt_path)
        self.prompt_prefix, self.answer_separator = build_prompt_prefix(prompt_cfg)
        self.schema = parse_schema(prompt_cfg)

        self.response_format: dict[str, Any] | None = None
        if self.use_structured_output:
            self.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "math_classification",
                    "schema": build_json_schema(self.schema),
                },
            }

        default_actions = {}
        for label in self.schema:
            if label in ("keep", "posttrain"):
                default_actions[label] = "keep"
            else:
                default_actions[label] = label
        self.label_actions: dict[str, str] = {
            **default_actions,
            **self.config.get("label_actions", {}),
        }

        log.info(
            "LLMClassifier: api=%s model=%s concurrency=%d target_phase=%r "
            "labels=%s prefix_len=%d max_content_chars=%d structured_output=%s",
            self.api_base, self.model, self.concurrency, self.target_phase,
            sorted(self.schema.keys()), len(self.prompt_prefix),
            self.max_content_chars,
            self.use_structured_output,
        )

    def process(
        self,
        records: list[CanonicalRecord],
        *,
        pipeline_artifacts: dict[str, Any] | None = None,
    ) -> DatasetProcessorResult:
        target = self.target_phase
        rows: list[dict[str, str]] = []
        row_idx: dict[str, int] = {}

        for i, rec in enumerate(records):
            if rec.training_phase == target:
                rows.append({
                    "record_id": rec.record_id,
                    "question": rec.question,
                    "answer": rec.raw_dataset_answer or "",
                })
                row_idx[rec.record_id] = i

        log.info(
            "LLMClassifier: total=%d target_phase=%r matched=%d",
            len(records), target, len(rows),
        )

        if not rows:
            return DatasetProcessorResult(kept_records=list(records))

        api_results = asyncio.run(
            _classify_batch(
                rows=rows,
                api_base=self.api_base,
                model=self.model,
                prompt_prefix=self.prompt_prefix,
                answer_separator=self.answer_separator,
                concurrency=self.concurrency,
                max_retries=self.max_retries,
                max_content_chars=self.max_content_chars,
                schema=self.schema,
                response_format=self.response_format,
            )
        )

        label_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}
        kept: list[CanonicalRecord] = list(records)
        proc_results: list[ProcessorResult] = []

        for rid, (label, tag, raw, attempts) in api_results.items():
            label_counts[label] = label_counts.get(label, 0) + 1
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            idx = row_idx[rid]
            rec = kept[idx]
            action = self.label_actions.get(label, label)

            if label == "api_error":
                rec.add_trace(
                    stage="clean", processor=self.name,
                    status="error", reason_code="api_error",
                    details={"attempts": attempts},
                )
                proc_results.append(ProcessorResult(
                    keep=True, record=rec, stage="clean",
                    processor=self.name, reason_code="api_error",
                ))
            elif action == "keep":
                rec.add_trace(
                    stage="clean", processor=self.name,
                    status="kept", reason_code=tag,
                    details={"attempts": attempts},
                )
                proc_results.append(ProcessorResult(
                    keep=True, record=rec, stage="clean",
                    processor=self.name, reason_code=tag,
                ))
            else:
                updated = rec.clone(training_phase=action, filter_tag=tag)
                status = "dropped" if action == "drop" else "routed"
                updated.add_trace(
                    stage="clean", processor=self.name,
                    status=status, reason_code=tag,
                    details={"raw": raw[:120], "attempts": attempts},
                )
                kept[idx] = updated
                proc_results.append(ProcessorResult(
                    keep=True, record=updated, stage="clean",
                    processor=self.name, reason_code=tag,
                ))

        log.info("LLMClassifier label dist: %s", label_counts)
        log.info("LLMClassifier tag   dist: %s", tag_counts)
        return DatasetProcessorResult(kept_records=kept, processor_results=proc_results)
