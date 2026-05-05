from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.core.io import write_canonical_records
from pipeline.core.schema import CanonicalRecord


@dataclass(slots=True)
class SourceConfig:
    name: str
    dataset_name: str
    input_path: str
    output_path: str
    source_split: str = "train"
    options: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceConfig":
        return cls(
            name=str(data.get("name", "")),
            dataset_name=str(data.get("dataset_name", "")),
            input_path=str(data.get("input_path", "")),
            output_path=str(data.get("output_path", "")),
            source_split=str(data.get("source_split", "train")),
            options=dict(data.get("options", {})),
        )

    @property
    def input(self) -> Path:
        return Path(self.input_path)

    @property
    def output(self) -> Path:
        return Path(self.output_path)


@dataclass(slots=True)
class IngestSummary:
    record_count: int
    output_paths: list[Path]


class BaseSourceAdapter(ABC):
    def __init__(self, config: SourceConfig) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> list[CanonicalRecord]:
        raise NotImplementedError

    def write(self) -> IngestSummary:
        records = self.load()
        write_canonical_records(records, self.config.output)
        return IngestSummary(record_count=len(records), output_paths=[self.config.output])
