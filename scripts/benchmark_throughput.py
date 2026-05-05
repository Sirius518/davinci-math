"""
Throughput benchmark for LLM classifier.

Tests multiple concurrency levels on a subset of data, reports:
  - wall time, req/sec, tokens/sec, avg latency, p50/p99, success rate

Usage:
    PYTHONPATH=src python scripts/benchmark_throughput.py \
        --data  data/processed/small_sample_test.parquet \
        --prompt configs/prompts/posttrain_filter.yaml \
        --api-base http://localhost:8000/v1 \
        --model gpt-oss-20b \
        --sample 3000 \
        --concurrency 32,64,128,256,512,1024
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

import yaml

TAG_RE = re.compile(r"\[(\w+)\]")


def build_prompt_prefix(cfg: dict) -> tuple[str, str]:
    parts = [cfg.get("instruction", "").strip()]
    parts.append("\n## Categories\n")
    for label, defn in cfg.get("category_definitions", {}).items():
        parts.append(f"**{label}**\n{str(defn).strip()}\n")
    output_fmt = cfg.get("output_format", "")
    if output_fmt:
        parts.append(f"## Output Format\n\n{output_fmt.strip()}\n")
    examples = cfg.get("examples", [])
    if examples:
        parts.append("## Examples\n")
        ans_sep = cfg.get("answer_separator", "\nAnswer: ")
        for ex in examples:
            parts.append(f"Question: {ex['question']}{ans_sep}{ex['answer']}")
            parts.append(f'{{"label": "{ex["label"]}", "tag": "{ex["tag"]}"}}\n')
    suffix = cfg.get("suffix", "Now classify the following sample.\n\nQuestion: ")
    parts.append(suffix)
    return "\n".join(parts), cfg.get("answer_separator", "\nAnswer: ")


def build_schema_and_format(cfg: dict) -> tuple[dict, dict]:
    schema = {}
    for label, defn in cfg.get("category_definitions", {}).items():
        tags = set(TAG_RE.findall(str(defn)))
        if "Tag:" in str(defn):
            for line in str(defn).splitlines():
                if line.strip().lower().startswith("tag:"):
                    t = line.split(":", 1)[1].strip()
                    if t:
                        tags.add(t)
        schema[label] = frozenset(tags) if tags else frozenset({label})

    all_labels = sorted(schema.keys())
    all_tags = sorted({t for ts in schema.values() for t in ts})
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "math_classification",
            "schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": all_labels},
                    "tag": {"type": "string", "enum": all_tags},
                },
                "required": ["label", "tag"],
            },
        },
    }
    return schema, response_format


async def _send_one(session, url, payload, semaphore):
    t0 = time.perf_counter()
    async with semaphore:
        try:
            async with session.post(url, json=payload, timeout=__import__("aiohttp").ClientTimeout(total=120)) as resp:
                data = await resp.json()
        except Exception as e:
            return {"ok": False, "latency": time.perf_counter() - t0, "error": str(e)}

    latency = time.perf_counter() - t0
    try:
        ch = data["choices"][0]
        content = ch["message"]["content"]
        finish = ch.get("finish_reason", "?")
        comp_tokens = data.get("usage", {}).get("completion_tokens", 0)
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        return {
            "ok": finish == "stop" and content is not None,
            "latency": latency,
            "comp_tokens": comp_tokens,
            "prompt_tokens": prompt_tokens,
            "finish": finish,
        }
    except (KeyError, IndexError):
        return {"ok": False, "latency": latency, "error": json.dumps(data)[:200]}


async def run_bench(
    rows: list[dict],
    prompt_prefix: str,
    answer_separator: str,
    response_format: dict,
    api_base: str,
    model: str,
    concurrency: int,
) -> dict:
    import aiohttp

    url = f"{api_base}/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 32, ttl_dns_cache=300)

    payloads = []
    for row in rows:
        full_prompt = prompt_prefix + row["question"] + answer_separator + row["answer"] + "\n"
        payloads.append({
            "model": model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 48,
            "response_format": response_format,
        })

    wall_start = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_send_one(session, url, p, semaphore) for p in payloads]
        results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - wall_start

    ok_results = [r for r in results if r["ok"]]
    fail_results = [r for r in results if not r["ok"]]
    latencies = [r["latency"] for r in ok_results]
    total_comp_tokens = sum(r.get("comp_tokens", 0) for r in ok_results)
    total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in ok_results)

    return {
        "concurrency": concurrency,
        "total": len(results),
        "success": len(ok_results),
        "failed": len(fail_results),
        "wall_time": wall_time,
        "req_per_sec": len(ok_results) / wall_time if wall_time > 0 else 0,
        "comp_tok_per_sec": total_comp_tokens / wall_time if wall_time > 0 else 0,
        "prompt_tok_per_sec": total_prompt_tokens / wall_time if wall_time > 0 else 0,
        "avg_latency": statistics.mean(latencies) if latencies else 0,
        "p50_latency": statistics.median(latencies) if latencies else 0,
        "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
        "fail_examples": [r.get("error", r.get("finish", "?")) for r in fail_results[:3]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="gpt-oss-20b")
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--concurrency", default="32,64,128,256,512,1024")
    args = parser.parse_args()

    concurrency_levels = [int(x.strip()) for x in args.concurrency.split(",")]

    with open(args.prompt) as f:
        cfg = yaml.safe_load(f)
    prompt_prefix, answer_separator = build_prompt_prefix(cfg)
    schema, response_format = build_schema_and_format(cfg)

    import pyarrow.parquet as pq
    table = pq.read_table(args.data)
    n = min(args.sample, table.num_rows)
    import random
    random.seed(42)
    indices = random.sample(range(table.num_rows), n)
    subset = table.take(indices)

    rows = []
    for i in range(subset.num_rows):
        rows.append({
            "question": str(subset.column("question")[i].as_py()),
            "answer": str(subset.column("raw_dataset_answer")[i].as_py() or ""),
        })

    print(f"{'='*90}")
    print(f"Throughput Benchmark: {n} samples, model={args.model}")
    print(f"Prompt prefix: {len(prompt_prefix)} chars")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"{'='*90}\n")

    # warmup: send 10 requests to populate KV cache
    print("Warming up KV cache (10 requests)...")
    asyncio.run(run_bench(rows[:10], prompt_prefix, answer_separator,
                          response_format, args.api_base, args.model, 10))
    print("Warmup done.\n")

    results = []
    for conc in concurrency_levels:
        print(f"--- Testing concurrency={conc} with {n} requests ---")
        r = asyncio.run(run_bench(rows, prompt_prefix, answer_separator,
                                  response_format, args.api_base, args.model, conc))
        results.append(r)
        print(f"  wall_time   : {r['wall_time']:.1f}s")
        print(f"  req/sec     : {r['req_per_sec']:.1f}")
        print(f"  comp_tok/s  : {r['comp_tok_per_sec']:.0f}")
        print(f"  prompt_tok/s: {r['prompt_tok_per_sec']:.0f}")
        print(f"  avg_latency : {r['avg_latency']*1000:.0f}ms")
        print(f"  p50_latency : {r['p50_latency']*1000:.0f}ms")
        print(f"  p99_latency : {r['p99_latency']*1000:.0f}ms")
        print(f"  success     : {r['success']}/{r['total']} ({100*r['success']/r['total']:.1f}%)")
        if r["fail_examples"]:
            print(f"  fail samples: {r['fail_examples']}")
        print()

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Conc':>6} | {'Wall(s)':>8} | {'Req/s':>8} | {'CompTok/s':>10} | "
          f"{'AvgLat(ms)':>10} | {'P50(ms)':>8} | {'P99(ms)':>8} | {'Success%':>8}")
    print("-" * 90)
    best = max(results, key=lambda r: r["req_per_sec"])
    for r in results:
        marker = " <-- BEST" if r is best else ""
        print(f"{r['concurrency']:>6} | {r['wall_time']:>8.1f} | {r['req_per_sec']:>8.1f} | "
              f"{r['comp_tok_per_sec']:>10.0f} | {r['avg_latency']*1000:>10.0f} | "
              f"{r['p50_latency']*1000:>8.0f} | {r['p99_latency']*1000:>8.0f} | "
              f"{100*r['success']/r['total']:>7.1f}%{marker}")
    print(f"\nOptimal concurrency: {best['concurrency']} ({best['req_per_sec']:.1f} req/s)")


if __name__ == "__main__":
    main()
