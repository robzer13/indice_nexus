"""FastAPI application exposing stock analysis services."""
from __future__ import annotations

from .app import app
from . import market_endpoints as _market_endpoints  # registers historical GET routes

__all__ = ["app"]
