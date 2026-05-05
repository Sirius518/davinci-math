from __future__ import annotations

from pathlib import Path

from pipeline.core.io import read_canonical_records, write_canonical_records
from pipeline.core.schema import CanonicalRecord
from pipeline.utils.jsonx import dumps as json_dumps
from pipeline.utils.jsonx import loads as json_loads


class Checkpoint:
    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.root = Path(checkpoint_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text(json_dumps({"completed": []}, indent=2), encoding="utf-8")

    def _index(self) -> dict[str, list[str]]:
        payload = json_loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"completed": []}
        return payload

    def _write_index(self, payload: dict[str, list[str]]) -> None:
        self.index_path.write_text(json_dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _step_path(self, step_name: str) -> Path:
        return self.root / f"{step_name}.parquet"

    def is_completed(self, step_name: str) -> bool:
        index = self._index()
        return step_name in set(index.get("completed", []))

    def load_output(self, step_name: str) -> list[CanonicalRecord]:
        path = self._step_path(step_name)
        if path.exists():
            return read_canonical_records(path)
        for candidate in sorted(self.root.glob("*.parquet"), reverse=True):
            if candidate.stem.endswith(step_name) or step_name in candidate.stem:
                return read_canonical_records(candidate)
        raise FileNotFoundError(f"No checkpoint data found for step '{step_name}' in {self.root}")

    def save_output(self, step_name: str, records: list[CanonicalRecord]) -> None:
        write_canonical_records(records, self._step_path(step_name))
        self.mark_completed(step_name)

    def mark_completed(self, step_name: str) -> None:
        """Mark a step as completed without writing a new parquet file."""
        index = self._index()
        completed = set(index.get("completed", []))
        completed.add(step_name)
        index["completed"] = sorted(completed)
        self._write_index(index)
