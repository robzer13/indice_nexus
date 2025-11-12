"""Minimal stub of pydantic for offline unit tests."""
from __future__ import annotations

from typing import Any, Dict


class BaseModel:
    def __init__(self, **data: Any) -> None:
        for key, value in self._defaults().items():
            if isinstance(value, (dict, list)):
                value = value.copy()
            setattr(self, key, value)
        for key, value in data.items():
            setattr(self, key, value)

    @classmethod
    def _defaults(cls) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        for key, value in cls.__dict__.items():
            if key.startswith("_") or callable(value):
                continue
            defaults[key] = value
        return defaults

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {key: getattr(self, key) for key in self._defaults().keys() if hasattr(self, key)}


def Field(default: Any = None, *, default_factory: Any | None = None, **_: Any) -> Any:
    if default_factory is not None:
        return default_factory()
    return default
