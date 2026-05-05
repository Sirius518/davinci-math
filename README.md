# Dataset Pipeline

A modular pipeline for ingesting, cleaning, deduplicating, decontaminating,
verifying, and rolling out math reasoning datasets at scale.

## Features

- **Source ingestion** into a single canonical record schema (`SCHEMA.md`).
- **Rule-based cleaning**: language, length, MCQ conversion, math-verify
  correctness checks, prompt-contamination filters, and more.
- **Exact and fuzzy (MinHash / LSH) deduplication** with optional model-judged
  pairs via SGLang or vLLM.
- **Benchmark decontamination** through embedding recall plus LLM judging.
- **Rollout-based correctness verification** with majority voting, GT
  reconciliation, and pass-ratio difficulty annotation.
- **CoT distillation hooks** and pass-ratio-based downstream filtering.
- **Checkpoint / resume** support across all stages, plus an incremental
  ingest path for adding new datasets without reprocessing the corpus.

## Installation

```bash
pip install -e .
```

This pulls in the runtime requirements listed in `requirements.txt`
(`pyarrow`, `datasketch`, `numba`, `tiktoken`, `xxhash`, `orjson`,
`aiohttp`, `faiss-cpu`, `math-verify`, ...).

## Quick Start

The CLI lives at `pipeline.cli` (also installed as the `pipeline` console
script when the package is installed):

```bash
# 1. Convert raw datasets into the canonical schema.
python -m pipeline.cli ingest --config configs/sources/data_same_format.yaml

# 2. Run a pipeline (cleaning, dedup, classification, ...).
python -m pipeline.cli run --config configs/pipelines/math_clean_pipeline_v2.yaml

# 3. Add a new dataset on top of an already-processed corpus.
python -m pipeline.cli add --config configs/pipelines/add_dataset_template.yaml
```

All paths inside the bundled YAML configs are written relative to the
project root. Edit them to point at your own data directories before
running.

## Repository Layout

```
src/pipeline/
    cli/                # argparse entry point (`python -m pipeline.cli ...`)
    core/               # canonical schema, registry, IO, checkpoint, runner
    sources/            # raw-format adapters (same_format, openreasoning, ...)
    ops/clean/          # rule-based cleaning processors
    ops/dedup/          # exact + MinHash/LSH dedup, model-judged dedup
    ops/decontaminate/  # benchmark decontamination (embedding + LLM judge)
    ops/evaluate/       # rollout, answer-judge, difficulty, pass ratio
    ops/llm/            # SGLang / vLLM async clients
    utils/              # hashing, JSON helpers, text utilities

configs/
    models/             # backend configs (SGLang, vLLM)
    runtime/            # runtime defaults (workers, retries, timeouts)
    sources/            # ingestion configs per raw format
    pipelines/          # full pipeline configs (cleaning, dedup, rollout, ...)
    prompts/            # judge / rollout / decontamination prompt templates

scripts/                # analysis, patching, and stats utilities
tests/                  # pytest test suite
SCHEMA.md               # canonical record schema reference
```

## Schema

The canonical record schema and the contracts of its nested fields
(`verification`, `distillation`, `decontamination`, `meta`, `trace`) are
documented in `SCHEMA.md`.

## Configuration

Pipelines are described as YAML, e.g.:

```yaml
name: my_pipeline
input_path: data/intermediate/my_shards
output_path: data/processed/my_corpus.parquet
checkpoint_dir: artifacts/checkpoints/my_pipeline

processors:
  - name: text_cleaning
  - name: language_filter
  - name: exact_dedup
  - name: fuzzy_dedup
    threshold: 0.8
    num_perm: 64
    bands: 8
    ngram_size: 3
```

The full list of available processor names is registered in
`src/pipeline/core/registry.py`.

## LLM Backends

Both SGLang and vLLM are supported. The model config files in
`configs/models/` declare a `backend`, `base_url`, and `model`. Any
processor that calls into an LLM accepts a `model_config` field pointing
at one of those YAML files.

```yaml
backend: sglang
base_url: http://127.0.0.1:30000
model: gpt-oss-120b
timeout_seconds: 120
max_retries: 3
cache_dir: ./artifacts/cache/sglang
```

Bring up the backend out-of-band (e.g. `python -m sglang.launch_server`
or `vllm serve`), then point the pipeline config at it.

## Running Tests

```bash
pip install pytest
PYTHONPATH=src pytest tests
```

## License

MIT — see `LICENSE`.
