"""Process-level memory and resource monitoring utilities."""
from __future__ import annotations

import os


def _log_stage2(message: str) -> None:
    print(f"[fuzzy_dedup] {message}", flush=True)


def _rss_bytes(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def _smaps_rollup_bytes(pid: int) -> tuple[int, int, int]:
    pss = 0
    private_dirty = 0
    private_clean = 0
    try:
        with open(f"/proc/{pid}/smaps_rollup", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Pss:"):
                    pss = int(line.split()[1]) * 1024
                elif line.startswith("Private_Dirty:"):
                    private_dirty = int(line.split()[1]) * 1024
                elif line.startswith("Private_Clean:"):
                    private_clean = int(line.split()[1]) * 1024
    except OSError:
        return 0, 0, 0
    return pss, private_dirty, private_clean


def _child_pids(pid: int) -> list[int]:
    try:
        with open(f"/proc/{pid}/task/{pid}/children", "r", encoding="utf-8") as handle:
            content = handle.read().strip()
    except OSError:
        return []
    if not content:
        return []
    return [int(part) for part in content.split() if part.isdigit()]


def _log_snapshot(phase: str, **extra: object) -> None:
    pid = os.getpid()
    child_pids = _child_pids(pid)
    child_rss_bytes = sum(_rss_bytes(child_pid) for child_pid in child_pids)
    pss_bytes, private_dirty_bytes, private_clean_bytes = _smaps_rollup_bytes(pid)
    child_pss_bytes = 0
    child_private_dirty_bytes = 0
    child_private_clean_bytes = 0
    for child_pid in child_pids:
        child_pss, child_private_dirty, child_private_clean = _smaps_rollup_bytes(child_pid)
        child_pss_bytes += child_pss
        child_private_dirty_bytes += child_private_dirty
        child_private_clean_bytes += child_private_clean
    payload = {
        "phase": phase,
        "pid": pid,
        "rss_gb": f"{_rss_bytes(pid) / (1024 ** 3):.2f}",
        "pss_gb": f"{pss_bytes / (1024 ** 3):.2f}",
        "private_dirty_gb": f"{private_dirty_bytes / (1024 ** 3):.2f}",
        "private_clean_gb": f"{private_clean_bytes / (1024 ** 3):.2f}",
        "child_count": len(child_pids),
        "child_rss_gb": f"{child_rss_bytes / (1024 ** 3):.2f}",
        "child_pss_gb": f"{child_pss_bytes / (1024 ** 3):.2f}",
        "child_private_dirty_gb": f"{child_private_dirty_bytes / (1024 ** 3):.2f}",
        "child_private_clean_gb": f"{child_private_clean_bytes / (1024 ** 3):.2f}",
    }
    payload.update({key: value for key, value in extra.items() if value is not None})
    _log_stage2("resource_snapshot " + " ".join(f"{key}={value}" for key, value in payload.items()))
