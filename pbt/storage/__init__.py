"""Storage backends for pbt runtime state and caching."""

from __future__ import annotations

from pbt.storage.base import StorageBackend
from pbt.storage.memory import MemoryStorageBackend

__all__ = ["MemoryStorageBackend", "SQLiteStorageBackend", "StorageBackend"]


def __getattr__(name: str):
    if name == "SQLiteStorageBackend":
        from pbt.storage.sqlite import SQLiteStorageBackend
        return SQLiteStorageBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
