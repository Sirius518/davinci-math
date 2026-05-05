from __future__ import annotations

from typing import Any

from pipeline.core.schema import DatasetProcessor, RecordProcessor


ProcessorType = type[RecordProcessor] | type[DatasetProcessor]
_REGISTRY: dict[str, ProcessorType] = {}


def register_processor(name: str):
    def decorator(cls: ProcessorType) -> ProcessorType:
        _REGISTRY[name] = cls
        return cls

    return decorator


def build_processor(name: str, config: dict[str, Any] | None = None) -> RecordProcessor | DatasetProcessor:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown processor: {name}. Available: {available}")
    cls = _REGISTRY[name]
    return cls(config or {})


def list_processors() -> list[str]:
    return sorted(_REGISTRY.keys())
