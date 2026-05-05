from __future__ import annotations

from pipeline.sources.base import BaseSourceAdapter, SourceConfig
from pipeline.sources.nemotron_math_v2 import NemotronMathV2Adapter
from pipeline.sources.openreasoning import OpenReasoningAdapter
from pipeline.sources.same_format import SameFormatAdapter


SOURCE_REGISTRY = {
    "openreasoning": OpenReasoningAdapter,
    "nemotron_math_v2": NemotronMathV2Adapter,
    "same_format": SameFormatAdapter,
}


def build_source_adapter(config: SourceConfig) -> BaseSourceAdapter:
    try:
        adapter_cls = SOURCE_REGISTRY[config.name]
    except KeyError as error:
        raise ValueError(f"Unsupported source adapter: {config.name}") from error
    return adapter_cls(config)
