"""Lightweight Streamlit stub so unit tests can import the dashboard module."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, List


DEFAULT_STATE: Dict[str, Any] = {
    "tickers": ["MC.PA"],
    "period": "1y",
    "interval": "1d",
    "price_column": "Close",
    "ml_horizon": 5,
    "button::Analyser": False,
    "button::Rafraîchir le cache": False,
    "button::Télécharger": False,
}

STATE: Dict[str, Any] = dict(DEFAULT_STATE)


def configure(**overrides: Any) -> None:
    STATE.update(overrides)


def reset() -> None:
    STATE.clear()
    STATE.update(DEFAULT_STATE)


def _log(_: str, *__: Any, **___: Any) -> None:
    return None


def _button(label: str, **__: Any) -> bool:
    return bool(STATE.get(f"button::{label}", False))


def _select(option_key: str, default: Any) -> Any:
    return STATE.get(option_key, default)


def multiselect(label: str, options: Iterable[str], default: Iterable[str] | None = None, **_: Any) -> List[str]:
    value = STATE.get("tickers")
    if value is None:
        return list(default or [])
    return list(value)


def text_input(label: str, value: str = "", **_: Any) -> str:
    return str(_select(label, value))


def selectbox(label: str, options: Iterable[Any], index: int = 0, **_: Any) -> Any:
    options = list(options)
    configured = STATE.get(label)
    if configured in options:
        return configured
    if 0 <= index < len(options):
        return options[index]
    return options[0] if options else None


def number_input(label: str, value: int, **_: Any) -> int:
    configured = STATE.get(label)
    if isinstance(configured, (int, float)):
        return int(configured)
    return int(value)


def line_chart(*_: Any, **__: Any) -> None:
    return None


def plotly_chart(*_: Any, **__: Any) -> None:
    return None


def dataframe(*_: Any, **__: Any) -> None:
    return None


def download_button(label: str, data: str | bytes, file_name: str, mime: str = "text/plain", **__: Any) -> bool:
    return bool(STATE.get(f"button::{label}", True))


@contextmanager
def expander(label: str, expanded: bool = False):
    yield None


class _Sidebar:
    def multiselect(self, label: str, options: Iterable[str], default: Iterable[str] | None = None, **kwargs: Any) -> List[str]:
        return multiselect(label, options, default=default, **kwargs)

    def selectbox(self, label: str, options: Iterable[Any], index: int = 0, **kwargs: Any) -> Any:
        return selectbox(label, options, index=index, **kwargs)

    def text_input(self, label: str, value: str = "", **kwargs: Any) -> str:
        return text_input(label, value=value, **kwargs)

    def number_input(self, label: str, value: int, **kwargs: Any) -> int:
        return number_input(label, value=value, **kwargs)

    def button(self, label: str, **kwargs: Any) -> bool:
        return _button(label, **kwargs)


sidebar = _Sidebar()


def set_page_config(*_: Any, **__: Any) -> None:
    return None


def title(*args: Any, **kwargs: Any) -> None:
    _log("title", *args, **kwargs)


def header(*args: Any, **kwargs: Any) -> None:
    _log("header", *args, **kwargs)


def subheader(*args: Any, **kwargs: Any) -> None:
    _log("subheader", *args, **kwargs)


def markdown(*args: Any, **kwargs: Any) -> None:
    _log("markdown", *args, **kwargs)


def write(*args: Any, **kwargs: Any) -> None:
    _log("write", *args, **kwargs)


def info(*args: Any, **kwargs: Any) -> None:
    _log("info", *args, **kwargs)


def warning(*args: Any, **kwargs: Any) -> None:
    _log("warning", *args, **kwargs)


def error(*args: Any, **kwargs: Any) -> None:
    _log("error", *args, **kwargs)


def success(*args: Any, **kwargs: Any) -> None:
    _log("success", *args, **kwargs)


def button(label: str, **kwargs: Any) -> bool:
    return _button(label, **kwargs)


def columns(sizes: Iterable[int]) -> List[Any]:
    return [None for _ in sizes]


__all__ = [
    "configure",
    "reset",
    "set_page_config",
    "title",
    "header",
    "subheader",
    "markdown",
    "write",
    "info",
    "warning",
    "error",
    "success",
    "button",
    "line_chart",
    "plotly_chart",
    "dataframe",
    "download_button",
    "sidebar",
    "expander",
    "columns",
]
