"""Durable conversation snapshots and one-process-at-a-time protection."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, BinaryIO


class Store:
    """Save an entire conversation in one SQLite transaction."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, timeout=5)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS conversations ("
                "name TEXT PRIMARY KEY, state_json TEXT NOT NULL, "
                "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )

    def load(self, name: str, default_provider: str = "gemini") -> dict[str, Any]:
        row = self._connection.execute(
            "SELECT state_json FROM conversations WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return {
                "provider": default_provider,
                "messages": [],
                "pending": None,
                "active": False,
            }
        state = json.loads(row[0])
        if not isinstance(state, dict):
            raise ValueError(f"Conversation {name!r} is not a JSON object")
        return state

    def save(self, name: str, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("Conversation state must be a dictionary")
        payload = json.dumps(state, ensure_ascii=False, allow_nan=False)
        with self._connection:
            self._connection.execute(
                "INSERT INTO conversations(name, state_json) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET state_json = excluded.state_json, "
                "updated_at = CURRENT_TIMESTAMP",
                (name, payload),
            )

    def close(self) -> None:
        self._connection.close()


class AppLock:
    """An OS-released, nonblocking lock shared by all conversations."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._file: BinaryIO | None = None

    def __enter__(self) -> AppLock:
        if self._file is not None:
            raise RuntimeError("This application lock is already held")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                # Windows locks a byte range; guarantee a byte exists first.
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(
                "Another agent instance is already running for this project. "
                "Close it before starting another."
            ) from exc
        self._file = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        handle, self._file = self._file, None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
