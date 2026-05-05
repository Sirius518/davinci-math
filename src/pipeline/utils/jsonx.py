from __future__ import annotations

from typing import Any, Callable, Iterable

import orjson  # type: ignore[reportMissingImports]


_Normalizer = Callable[[Any], Any]
_MISSING = object()


class _LazyJsonDict(dict[str, Any]):
    __slots__ = ("_raw", "_normalizer")

    def __init__(self, raw: str | bytes | bytearray | None, normalizer: _Normalizer | None = None) -> None:
        super().__init__()
        self._raw = raw
        self._normalizer = normalizer

    def _resolve(self) -> None:
        raw = self._raw
        if raw is None:
            return
        self._raw = None
        if not raw:
            return
        payload = orjson.loads(raw)
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise TypeError(f"Expected JSON object payload, got {type(payload).__name__}")
        normalizer = self._normalizer
        if normalizer is not None:
            payload = {key: normalizer(value) for key, value in payload.items()}
        super().update(payload)

    def __getitem__(self, key: str) -> Any:
        self._resolve()
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._resolve()
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._resolve()
        super().__delitem__(key)

    def __iter__(self):
        self._resolve()
        return super().__iter__()

    def __len__(self) -> int:
        self._resolve()
        return super().__len__()

    def __contains__(self, key: object) -> bool:
        self._resolve()
        return super().__contains__(key)

    def __bool__(self) -> bool:
        self._resolve()
        return super().__len__() > 0

    def __repr__(self) -> str:
        self._resolve()
        return super().__repr__()

    def __eq__(self, other: object) -> bool:
        self._resolve()
        return super().__eq__(other)

    def get(self, key: str, default: Any = None) -> Any:
        self._resolve()
        return super().get(key, default)

    def keys(self):
        self._resolve()
        return super().keys()

    def items(self):
        self._resolve()
        return super().items()

    def values(self):
        self._resolve()
        return super().values()

    def update(self, *args: Any, **kwargs: Any) -> None:
        self._resolve()
        super().update(*args, **kwargs)

    def clear(self) -> None:
        self._resolve()
        super().clear()

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        self._resolve()
        if default is _MISSING:
            return super().pop(key)
        return super().pop(key, default)

    def popitem(self):
        self._resolve()
        return super().popitem()

    def setdefault(self, key: str, default: Any = None) -> Any:
        self._resolve()
        return super().setdefault(key, default)

    def copy(self) -> dict[str, Any]:
        self._resolve()
        return dict(super().items())


class _LazyTraceList(list[Any]):
    __slots__ = ("_raw", "_normalizer")

    def __init__(self, raw: str | bytes | bytearray | None, normalizer: _Normalizer | None = None) -> None:
        super().__init__()
        self._raw = raw
        self._normalizer = normalizer

    def _resolve(self) -> None:
        raw = self._raw
        if raw is None:
            return
        self._raw = None
        if not raw:
            return
        payload = orjson.loads(raw)
        if payload is None:
            payload = []
        if not isinstance(payload, list):
            raise TypeError(f"Expected JSON array payload, got {type(payload).__name__}")
        normalizer = self._normalizer
        if normalizer is not None:
            payload = [normalizer(item) for item in payload]
        super().extend(payload)

    def __getitem__(self, index):
        self._resolve()
        return super().__getitem__(index)

    def __setitem__(self, index, value) -> None:
        self._resolve()
        super().__setitem__(index, value)

    def __delitem__(self, index) -> None:
        self._resolve()
        super().__delitem__(index)

    def __iter__(self):
        self._resolve()
        return super().__iter__()

    def __len__(self) -> int:
        self._resolve()
        return super().__len__()

    def __contains__(self, item: object) -> bool:
        self._resolve()
        return super().__contains__(item)

    def __bool__(self) -> bool:
        self._resolve()
        return super().__len__() > 0

    def __repr__(self) -> str:
        self._resolve()
        return super().__repr__()

    def __eq__(self, other: object) -> bool:
        self._resolve()
        return super().__eq__(other)

    def append(self, value: Any) -> None:
        self._resolve()
        super().append(value)

    def extend(self, values: Iterable[Any]) -> None:
        self._resolve()
        super().extend(values)

    def insert(self, index: int, value: Any) -> None:
        self._resolve()
        super().insert(index, value)

    def clear(self) -> None:
        self._resolve()
        super().clear()

    def pop(self, index: int = -1) -> Any:
        self._resolve()
        return super().pop(index)

    def remove(self, value: Any) -> None:
        self._resolve()
        super().remove(value)

    def copy(self) -> list[Any]:
        self._resolve()
        return list(super().__iter__())

    def index(self, value: Any, start: int = 0, stop: int = 9223372036854775807) -> int:
        self._resolve()
        return super().index(value, start, stop)

    def count(self, value: Any) -> int:
        self._resolve()
        return super().count(value)

    def sort(self, *, key: Callable[[Any], Any] | None = None, reverse: bool = False) -> None:
        self._resolve()
        super().sort(key=key, reverse=reverse)

    def reverse(self) -> None:
        self._resolve()
        super().reverse()


def _strip_surrogates(obj: Any) -> Any:
    """Recursively replace surrogate characters in strings."""
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, dict):
        return {_strip_surrogates(k): _strip_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_surrogates(item) for item in obj]
    return obj


def dumps(
    value: Any,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    indent: int | None = None,
) -> str:
    del ensure_ascii  # orjson always emits UTF-8 text.
    options = 0
    if sort_keys:
        options |= orjson.OPT_SORT_KEYS
    if indent == 2:
        options |= orjson.OPT_INDENT_2
    return orjson.dumps(_strip_surrogates(value), option=options).decode("utf-8")


def loads(payload: str | bytes | bytearray) -> Any:
    return orjson.loads(payload)
