"""Dataset-tier priority resolution for deduplication.

Maps each record's dataset_name to a numeric tier so that Union-Find and
exact-dedup keep the record from the highest-priority (lowest tier number)
dataset when duplicates are found.
"""
from __future__ import annotations

from pipeline.core.schema import CanonicalRecord


def _resolve_dataset_tier(
    dataset_name: str,
    exact_map: dict[str, int],
    prefix_keys: list[tuple[str, int]],
) -> int | None:
    """Resolve a dataset_name to a tier via exact match, then longest-prefix match."""
    tier = exact_map.get(dataset_name)
    if tier is not None:
        return tier
    best_tier: int | None = None
    best_len = 0
    for prefix, prefix_tier in prefix_keys:
        if dataset_name.startswith(prefix) and len(prefix) > best_len:
            best_tier = prefix_tier
            best_len = len(prefix)
    return best_tier


def build_record_priority_map(
    records: list[CanonicalRecord],
    tier_config: dict[int, list[str]],
) -> dict[str, int]:
    """Build a record_id -> tier mapping from records and a tier configuration.

    ``tier_config`` maps tier numbers (1 = highest priority) to lists of
    dataset name patterns.  Each pattern is first tried as an exact match
    against a record's ``dataset_name``; if that fails, it is tried as a
    prefix match (longest-prefix wins).

    Raises ``ValueError`` if any record has a ``dataset_name`` that cannot
    be resolved to a tier.
    """
    exact_map: dict[str, int] = {}
    prefix_keys: list[tuple[str, int]] = []
    for tier, names in tier_config.items():
        for name in names:
            exact_map[name] = int(tier)
            prefix_keys.append((name, int(tier)))
    prefix_keys.sort(key=lambda item: -len(item[0]))

    name_to_tier: dict[str, int] = {}
    record_map: dict[str, int] = {}
    unresolved: set[str] = set()

    for record in records:
        rid = record.record_id
        if rid in record_map:
            continue
        dn = record.dataset_name
        if dn in name_to_tier:
            record_map[rid] = name_to_tier[dn]
            continue
        tier = _resolve_dataset_tier(dn, exact_map, prefix_keys)
        if tier is None:
            unresolved.add(dn)
            continue
        name_to_tier[dn] = tier
        record_map[rid] = tier

    if unresolved:
        raise ValueError(
            f"Unknown dataset_name(s) not in dataset_priority config: "
            f"{sorted(unresolved)}"
        )
    return record_map


def parse_tier_config(raw: dict | None) -> dict[int, list[str]] | None:
    """Parse the ``dataset_priority`` section from a processor config dict.

    Returns ``None`` if not configured (priority-aware dedup is disabled).
    YAML keys may arrive as int or str; this normalises them.
    """
    if not raw:
        return None
    result: dict[int, list[str]] = {}
    for key, names in raw.items():
        tier = int(key)
        if not isinstance(names, list):
            raise ValueError(f"dataset_priority tier {tier} must be a list, got {type(names).__name__}")
        result[tier] = [str(n) for n in names]
    return result
