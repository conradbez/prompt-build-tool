"""Storage backends for promptmap runtime state and caching."""

from __future__ import annotations

from promptmap.storage.base import StorageBackend
from promptmap.storage.memory import MemoryStorageBackend

__all__ = ["MemoryStorageBackend", "SQLiteStorageBackend", "StorageBackend"]


def __getattr__(name: str):
    if name == "SQLiteStorageBackend":
        from promptmap.storage.sqlite import SQLiteStorageBackend
        return SQLiteStorageBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
