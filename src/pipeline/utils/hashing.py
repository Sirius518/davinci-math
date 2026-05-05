from __future__ import annotations

import hashlib
import xxhash  # type: ignore[reportMissingImports]


def fast_text_hash(value: str) -> str:
    return xxhash.xxh3_128_hexdigest(value.encode("utf-8"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_int_hash(value: str, *, seed: int = 0) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)
