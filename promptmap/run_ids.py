"""
Content hashes that identify what a run executed.

Three IDs, each a SHA256 hex digest of content, never of time or a UUID — so
two runs of the same inputs share them:

``template_hash``
    One model's raw ``.prompt`` source, before Jinja.  Stored on every
    ``model_results`` row; it is also what gates ``{{ previous }}`` (see
    :meth:`promptmap.executor.run_context.RunContext.previous_for`).

``map_hash``
    The whole project end to end: every model's name and source, the global
    instruction, and every file under the tests directory.  The same for a
    ``--select`` run as for a full one.

``run_hash``
    ``map_hash`` plus the template hashes of the models this run executed,
    the promptdata and the promptfiles' contents.  Two runs with the same
    ``run_hash`` were asked the same question.

The two run-level IDs are stored on the ``runs`` row by
:func:`record_run_ids`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from promptmap.model_spec import ModelSpec


def template_hash(source: str) -> str:
    """The hash of one model's raw template source."""
    return hashlib.sha256(source.encode()).hexdigest()


def map_hash(
    models: Iterable[ModelSpec],
    tests_dir: str | Path | None = None,
    global_instruction: str | None = None,
) -> str:
    """The hash of every prompt, the global instruction and every test file."""
    digest = hashlib.sha256()
    for spec in sorted(models, key=lambda s: s.name):
        _update(digest, "model", spec.name, template_hash(spec.source))
    _update(digest, "global", global_instruction or "")
    root = Path(tests_dir) if tests_dir is not None else None
    if root is not None and root.is_dir():
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            content = hashlib.sha256(path.read_bytes()).hexdigest()
            _update(digest, "test", path.relative_to(root).as_posix(), content)
    return digest.hexdigest()


def run_hash(
    map_id: str,
    models: Iterable[ModelSpec],
    promptdata: Mapping[str, Any] | None = None,
    promptfiles: Mapping[str, Any] | None = None,
) -> str:
    """The hash of the map, the models run, and the run's inputs."""
    digest = hashlib.sha256()
    _update(digest, "map", map_id)
    for spec in sorted(models, key=lambda s: s.name):
        _update(digest, "model", spec.name, template_hash(spec.source))
    _update(digest, "promptdata", json.dumps(promptdata or {}, sort_keys=True, default=str))
    for name in sorted(promptfiles or {}):
        _update(digest, "promptfile", name, _file_hash(promptfiles[name]))
    return digest.hexdigest()


def record_run_ids(
    storage: Any,
    run_id: str,
    all_models: Iterable[ModelSpec],
    run_models: Iterable[ModelSpec],
    tests_dir: str | Path | None = None,
    global_instruction: str | None = None,
    promptdata: Mapping[str, Any] | None = None,
    promptfiles: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Compute ``(map_hash, run_hash)`` and store them on the run's row.

    Storing is optional for a backend: one without ``record_run_ids`` still
    gets the hashes returned.
    """
    map_id = map_hash(all_models, tests_dir, global_instruction)
    run_id_hash = run_hash(map_id, run_models, promptdata, promptfiles)
    record = getattr(storage, "record_run_ids", None)
    if record is not None:
        record(run_id, map_hash=map_id, run_hash=run_id_hash)
    return map_id, run_id_hash


def _update(digest: Any, *parts: str) -> None:
    # Length-prefixed, so no two different part lists hash the same bytes.
    for part in parts:
        data = part.encode()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)


def _file_hash(handle: Any) -> str:
    """Hash a promptfile given as a path or a binary file-like object."""
    if isinstance(handle, (str, Path)):
        return hashlib.sha256(Path(handle).read_bytes()).hexdigest()
    known = getattr(handle, "sha256", None)  # a File from an upstream model
    if isinstance(known, str):
        return known
    try:
        position = handle.tell()
        data = handle.read()
        handle.seek(position)
    except Exception:  # noqa: BLE001 — an unreadable handle hashes by identity
        return repr(handle)
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode()).hexdigest()
