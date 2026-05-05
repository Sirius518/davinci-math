from __future__ import annotations

import re

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor

_PHANTOM_REFERENCE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"as shown in the figure",
        r"refer to the diagram",
        r"the figure (?:below|above)",
        r"see figure\s*\d+",
        r"in the picture",
        r"the table (?:below|above)",
        r"in the table",
        r"see table\s*\d+",
        r"from the chart",
        r"the graph is shown",
        r"the graph of .* is shown",
        r"shown in the graph",
        r"as illustrated",
        r"the image shows",
        r"see the drawing",
    ]
]


def has_phantom_reference(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PHANTOM_REFERENCE_PATTERNS)


@register_processor("phantom_reference_filter")
class PhantomReferenceFilterProcessor(RecordProcessor):
    name = "phantom_reference_filter"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        if record.training_phase:
            record.add_trace(
                stage="clean",
                processor=self.name,
                status="skipped",
                details={"training_phase": record.training_phase},
            )
            return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
        if has_phantom_reference(record.question):
            updated = record.clone(training_phase="drop", filter_tag="phantom_reference")
            updated.add_trace(stage="clean", processor=self.name, status="dropped", reason_code="phantom_reference")
            return ProcessorResult(
                keep=True,
                record=updated,
                stage="clean",
                processor=self.name,
                reason_code="phantom_reference",
            )
        record.add_trace(stage="clean", processor=self.name, status="kept")
        return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
