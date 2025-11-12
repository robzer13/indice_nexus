"""Minimal yfinance stub ensuring modules import offline."""

class Ticker:  # pragma: no cover - replaced by mocks in tests
    def __init__(self, *_: object, **__: object) -> None:
        raise RuntimeError("yfinance stub should be patched during tests")
