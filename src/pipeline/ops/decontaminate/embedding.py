from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import aiohttp
import numpy as np

log = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metadata_path(checkpoint_path: str | Path) -> Path:
    target = Path(checkpoint_path)
    return target.with_suffix(f"{target.suffix}.meta.json")


def save_embedding_checkpoint(
    checkpoint_path: str | Path,
    embeddings: np.ndarray,
    *,
    model: str,
    num_texts: int,
) -> None:
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, embeddings)
    metadata = {
        "model": model,
        "num_texts": int(num_texts),
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "dtype": str(embeddings.dtype),
        "created_at": _utc_now(),
    }
    _metadata_path(target).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_embedding_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: str,
    num_texts: int,
) -> np.ndarray | None:
    target = Path(checkpoint_path)
    meta_path = _metadata_path(target)
    if not target.exists() or not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if str(metadata.get("model", "")) != model:
        return None
    if int(metadata.get("num_texts", -1)) != num_texts:
        return None
    try:
        loaded = np.load(target, allow_pickle=False)
    except OSError:
        return None
    if loaded.ndim != 2 or loaded.shape[0] != num_texts:
        return None
    return np.asarray(loaded, dtype=np.float32)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings.astype(np.float32, copy=False)
    output = np.asarray(embeddings, dtype=np.float32).copy()
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    output /= norms
    return output


async def embed_batch(
    texts: Sequence[str],
    *,
    api_base: str,
    model: str,
    session: aiohttp.ClientSession,
    api_key: str = "",
    timeout_seconds: int = 120,
    encoding_format: str = "float",
    extra_body: dict[str, Any] | None = None,
) -> np.ndarray:
    """Call an OpenAI-compatible /v1/embeddings endpoint (e.g. SGLang).

    ``encoding_format`` is sent in the payload so that servers like SGLang
    return plain float arrays rather than base64.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    payload: dict[str, Any] = {
        "model": model,
        "input": list(texts),
        "encoding_format": encoding_format,
    }
    if extra_body:
        payload.update(extra_body)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{api_base.rstrip('/')}/embeddings"
    async with session.post(
        url,
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=timeout_seconds),
    ) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"embedding status={response.status} body={body[:400]}")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid embedding response: {body[:400]}") from exc
    items = parsed.get("data")
    if not isinstance(items, list):
        raise RuntimeError(f"embedding response missing data list: {body[:400]}")
    ordered = sorted(items, key=lambda item: int(item.get("index", 0)))
    vectors: list[list[float]] = []
    for item in ordered:
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("embedding item missing vector")
        vectors.append([float(value) for value in embedding])
    if len(vectors) != len(texts):
        raise RuntimeError(f"embedding batch size mismatch expected={len(texts)} got={len(vectors)}")
    return np.asarray(vectors, dtype=np.float32)


async def _embed_batch_with_retry(
    batch_index: int,
    texts: Sequence[str],
    *,
    api_base: str,
    model: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_key: str,
    timeout_seconds: int,
    max_retries: int,
    encoding_format: str,
    extra_body: dict[str, Any] | None,
) -> tuple[int, np.ndarray]:
    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                result = await embed_batch(
                    texts,
                    api_base=api_base,
                    model=model,
                    session=session,
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                    encoding_format=encoding_format,
                    extra_body=extra_body,
                )
                return batch_index, result
            except Exception as exc:
                log.warning(
                    "embedding batch=%d attempt=%d/%d failed: %s",
                    batch_index,
                    attempt,
                    max_retries,
                    exc,
                )
        await asyncio.sleep(min(2**attempt, 10))
    raise RuntimeError(f"embedding batch {batch_index} failed after {max_retries} attempts")


async def embed_all(
    texts: Sequence[str],
    *,
    api_base: str,
    model: str,
    batch_size: int = 256,
    concurrency: int = 16,
    api_key: str = "",
    timeout_seconds: int = 120,
    max_retries: int = 3,
    encoding_format: str = "float",
    extra_body: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    progress_name: str = "embeddings",
) -> np.ndarray:
    if checkpoint_path is not None:
        checkpoint = load_embedding_checkpoint(checkpoint_path, model=model, num_texts=len(texts))
        if checkpoint is not None:
            log.info("%s loaded from checkpoint: %s", progress_name, checkpoint_path)
            return checkpoint

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    batches = [list(texts[start : start + batch_size]) for start in range(0, len(texts), batch_size)]
    semaphore = asyncio.Semaphore(max(1, concurrency))
    connector = aiohttp.TCPConnector(limit=max(8, concurrency + 8), ttl_dns_cache=300)
    started_at = time.perf_counter()
    completed = 0
    total = len(batches)
    results: list[np.ndarray | None] = [None] * total

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                _embed_batch_with_retry(
                    index,
                    batch,
                    api_base=api_base,
                    model=model,
                    session=session,
                    semaphore=semaphore,
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    encoding_format=encoding_format,
                    extra_body=extra_body,
                )
            )
            for index, batch in enumerate(batches)
        ]
        for task in asyncio.as_completed(tasks):
            batch_index, batch_embeddings = await task
            results[batch_index] = batch_embeddings
            completed += 1
            if completed == total or completed % max(1, min(20, total)) == 0:
                elapsed = time.perf_counter() - started_at
                log.info("%s progress %d/%d batches elapsed=%.1fs", progress_name, completed, total, elapsed)

    arrays = [item for item in results if item is not None]
    if len(arrays) != total:
        raise RuntimeError(f"{progress_name} missing batches: expected {total}, got {len(arrays)}")
    merged = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    if checkpoint_path is not None:
        save_embedding_checkpoint(checkpoint_path, merged, model=model, num_texts=len(texts))
        log.info("%s checkpoint saved: %s", progress_name, checkpoint_path)
    return merged
