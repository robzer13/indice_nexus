"""Persistence helpers for behavioural analysis."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional, Sequence

from .schemas import BehaviorAnalysis, BehaviorPersistRecord

PersistenceMode = Literal["jsonl", "sqlite", "none"]
_DEFAULT_PATH = Path("out/behavior/behavior_records.jsonl")


class BehaviourStore:
    """Simple persistence layer for behaviour analyses."""

    def __init__(self, *, mode: PersistenceMode = "jsonl", path: Path | None = None) -> None:
        self.mode = mode
        self.path = path or _DEFAULT_PATH
        if self.mode == "sqlite":
            self.path = path or Path("out/behavior/behavior.db")
        if self.mode != "none":
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        tickers: Sequence[str],
        analysis: BehaviorAnalysis,
        context: dict | None = None,
    ) -> None:
        if self.mode == "none":
            return
        record = BehaviorPersistRecord(
            timestamp=datetime.now(timezone.utc),
            tickers=list(tickers),
            analysis=analysis,
            context=context or {},
        )
        if self.mode == "jsonl":
            _append_jsonl(self.path, record)
        elif self.mode == "sqlite":
            _append_sqlite(self.path, record)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported persistence mode: {self.mode}")

    def iter_records(self) -> Iterator[BehaviorPersistRecord]:
        if self.mode == "jsonl":
            yield from _iter_jsonl(self.path)
        elif self.mode == "sqlite":
            yield from _iter_sqlite(self.path)
        else:
            return iter(())


def _append_jsonl(path: Path, record: BehaviorPersistRecord) -> None:
    payload = {
        "timestamp": record.timestamp.isoformat(),
        "tickers": record.tickers,
        "analysis": record.analysis.dict(),
        "context": record.context,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _iter_jsonl(path: Path) -> Iterator[BehaviorPersistRecord]:
    if not path.exists():
        return iter(())
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            yield BehaviorPersistRecord(
                timestamp=datetime.fromisoformat(data["timestamp"]),
                tickers=list(data.get("tickers", [])),
                analysis=BehaviorAnalysis(**data.get("analysis", {})),
                context=data.get("context", {}),
            )


def _append_sqlite(path: Path, record: BehaviorPersistRecord) -> None:
    try:
        import sqlite3
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SQLite backend unavailable") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS behavior_records (
                timestamp TEXT,
                tickers TEXT,
                analysis TEXT,
                context TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO behavior_records VALUES (?, ?, ?, ?)",
            (
                record.timestamp.isoformat(),
                json.dumps(record.tickers),
                json.dumps(record.analysis.dict()),
                json.dumps(record.context),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _iter_sqlite(path: Path) -> Iterator[BehaviorPersistRecord]:
    try:
        import sqlite3
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("SQLite backend unavailable") from exc
    if not path.exists():
        return iter(())
    conn = sqlite3.connect(path)
    try:
        cursor = conn.execute("SELECT timestamp, tickers, analysis, context FROM behavior_records")
        for timestamp, tickers, analysis, context in cursor:
            yield BehaviorPersistRecord(
                timestamp=datetime.fromisoformat(timestamp),
                tickers=list(json.loads(tickers)),
                analysis=BehaviorAnalysis(**json.loads(analysis)),
                context=json.loads(context),
            )
    finally:
        conn.close()


__all__ = ["BehaviourStore", "PersistenceMode"]
