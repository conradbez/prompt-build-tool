"""promptmap — dbt-inspired LLM prompt orchestration."""

from __future__ import annotations

import re
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Awaitable, Callable, Mapping

from promptmap.executor.parser_initial import (
    UnknownConfigKeyWarning,
    known_config_keys,
    register_config_keys,
)
from promptmap.classifier import classifier_cache_check, systemone_classifier
from promptmap.executor.graph import UnknownModelTypeError
from promptmap.model_spec import ModelSpec
from promptmap.model_types import (
    ModelCall,
    ModelKind,
    get_model_kind,
    known_model_kinds,
    model_kind,
    register_model_kind,
)
from promptmap.files import (
    BlobStore,
    Dir,
    File,
    MemoryBlobStore,
    Output,
    SQLiteBlobStore,
)
from promptmap.secrets import Secrets
from promptmap.storage.base import StorageBackend
from promptmap.tokens import LLMResult
from promptmap.executor.executor import ModelRunResult
from promptmap.executor.parser_model import PreviousRun
from promptmap.types import PromptFile, PromptModelsDict

# Registers the built-in kinds ("", template, execute_python, agent).  Imported here,
# after the registry exists, so that every entry point into promptmap has them without
# the registry needing a lazy-import hook.
import promptmap.executor.builtin_kinds  # noqa: E402,F401

__all__ = [
    "run",
    "async_run",
    "ModelSpec",
    "ModelKind",
    "ModelCall",
    "model_kind",
    "register_model_kind",
    "get_model_kind",
    "known_model_kinds",
    "register_config_keys",
    "known_config_keys",
    "UnknownConfigKeyWarning",
    "UnknownModelTypeError",
    "systemone_classifier",
    "classifier_cache_check",
    "ModelStatus",
    "ModelError",
    "PromptModelsDict",
    "PromptFile",
    "StorageBackend",
    "Secrets",
    "File",
    "Dir",
    "Output",
    "LLMResult",
    "ModelRunResult",
    "PreviousRun",
    "BlobStore",
    "MemoryBlobStore",
    "SQLiteBlobStore",
    "__version__",
]


def _version_from_pyproject() -> str:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    contents = pyproject.read_text(encoding="utf-8")
    project_block = re.search(r"(?ms)^\[project\]\n(.*?)(?:^\[|\Z)", contents)
    if project_block is None:
        raise RuntimeError("Could not find [project] table in pyproject.toml")

    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', project_block.group(1), re.MULTILINE)
    if match is None:
        raise RuntimeError("Could not find project.version in pyproject.toml")
    return match.group(1)


def _detect_version() -> str:
    try:
        return package_version("promptmap")
    except PackageNotFoundError:
        return _version_from_pyproject()


__version__ = _detect_version()


class ModelStatus(Enum):
    """Returned in the promptmap.run() dict for models that produced no LLM output."""
    SKIPPED = "skipped"          # upstream dependency failed → model was not run
    ERROR = "error"              # LLM call or render raised an exception (legacy)


class ModelError:
    """Returned by promptmap.run() when a model fails; carries the exception message."""
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return f"error: {self.message}"


async def async_run(
    map_dir: str = "map",
    models_from_dict: PromptModelsDict | dict[str, str] | None = None,
    select: list[str] | None = None,
    llm_call: Callable[[str], str | Awaitable[str]] | None = None,
    rag_call: Callable[..., list] | None = None,
    verbose: bool = True,
    promptdata: dict | None = None,
    promptfiles: dict[str, PromptFile] | None = None,
    validation_dir: str | None = "validation",
    tests_dir: str | None = "tests",
    storage_backend: StorageBackend | None = None,
    global_instruction: str | Callable[[], str] | None = None,
    blob_store: BlobStore | None = None,
    on_model_start: Callable[[str], None] | None = None,
    on_model_done: Callable[[ModelRunResult], None] | None = None,
    secrets: Mapping[str, str] | None = None,
    cache_check: Callable[..., bool | Awaitable[bool]] | None = None,
):
    """
    Execute prompt models as a Python library call.

    Parameters
    ----------
    map_dir:
        Path to the directory containing *.prompt files.
    select:
        Optional list of model names to run. All upstream dependencies are
        also executed fresh; the prompt cache makes unchanged nodes instant.
    llm_call:
        Optional function ``(prompt: str) -> str`` to use as the LLM backend.
        Falls back to ``llm_call`` in client.py alongside *map_dir*; with
        neither, the run fails with a FileNotFoundError naming the fix.
    rag_call:
        Optional function ``(*args) -> list | str`` to back
        ``return_list_RAG_results()`` in templates.
        Falls back to ``map/rag.py::do_RAG`` if present.
    verbose:
        Print a dbt-style progress log to stdout (default: True).
    promptdata:
        Optional dict of runtime variables. Access them in templates via
        ``{{ promptdata('key') }}``.
    promptfiles:
        Optional dict mapping name → file reference. Models that declare
        ``promptfiles: name`` in their config block will receive these as a
        list passed to ``llm_call(prompt, files=[…])``.
        Values may be a **string path**, a :class:`pathlib.Path`, or any
        **binary file-like object** (``open(..., "rb")``, ``io.BytesIO``,
        etc.) — whatever your ``llm_call`` implementation accepts.
    tests_dir:
        The tests directory, hashed into the run's ``map_hash`` so a changed
        test changes the map's ID (see :mod:`promptmap.run_ids`).  Ignored
        for ``models_from_dict`` runs.
    global_instruction:
        Optional prompt text rendered into **every** model prompt — promptmap's
        analogue of dbt's query-comment.  Either a string or a zero-argument
        callable returning one.  Falls back to a ``global.prompt`` file
        alongside ``map_dir`` when not given (file-based runs only).

        The text is itself a Jinja template rendered with each model's own
        context, so ``promptdata()``, ``model.name`` and ``was_skipped()``
        work inside it; ``ref()`` does not, since that would make every model
        depend on the referenced one.  Reference ``{{ prompt }}`` to place the
        model body explicitly, otherwise the instruction is prepended.

        Individual models opt out with
        ``{{ config(global_instruction=False) }}``, and ``execute_python``
        models never receive it.
    blob_store:
        Where the bytes of file outputs (:class:`File`, :class:`Dir`,
        :class:`Output`) are kept.  Falls back to ``blob_store`` in client.py,
        then to the storage backend's own store (the SQLite database).
    on_model_start:
        Optional callback ``(model_name: str) -> None`` fired just before a
        model starts executing.  Use it for live progress in your own code.
    on_model_done:
        Optional callback ``(result: ModelRunResult) -> None`` fired when a
        model finishes, whether it succeeded, errored or was skipped.
        ``result.status`` is ``"success"``, ``"error"`` or ``"skipped"``.
    secrets:
        Optional dict of run-scoped credentials (API keys, tokens).  Model
        kinds read them as ``call.secrets["NAME"]``; an ``llm_call`` that
        declares a ``secrets`` parameter receives them as well, so one
        ``llm_call`` can serve a different key per run.  They are never
        rendered into templates, part of the cache key, stored, or shown in a
        repr — changing a key's value still gets cache hits.
    cache_check:
        Optional function ``(prompt: str, previous: PreviousRun) -> bool``
        asked on an exact prompt-cache miss whether the model's last
        successful result (``previous.prompt``, ``previous.output``) is still
        good enough for the new *prompt*.  Return True to reuse it as a cache
        hit — e.g. a cheap judge model deciding whether the prompt changed
        meaningfully.  Sync or async; may also declare ``config``,
        ``secrets`` and ``model`` parameters.  Falls back to ``cache_check``
        in client.py.

    Returns
    -------
    ``dict`` keyed by model name.  Each value is the model's output string —
    or, for a model that produced files, its :class:`File` / :class:`Dir` /
    :class:`Output` value (or the dict/list holding them) — or
    :class:`ModelStatus.SKIPPED` when an upstream model failed, or a
    :class:`ModelError` carrying the message when that model itself failed.

    The full per-model detail — rendered prompt, timing, cache hit — is not in
    this return value; it is written to storage and read back by ``promptmap test``
    and ``promptmap docs``.
    """
    import time
    from datetime import datetime

    from promptmap.executor.executor import execute_run, ModelRunResult
    from promptmap.executor.graph import (
        load_models,
        build_models_from_dict,
        build_dag,
    )
    if storage_backend is None:
        from promptmap.storage.sqlite import SQLiteStorageBackend
        storage_backend = SQLiteStorageBackend()

    console = None
    if verbose:
        from rich.console import Console
        console = Console(highlight=False, soft_wrap=True)

    def ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def log(msg: str) -> None:
        if verbose:
            console.print(f"[dim]{ts()}[/dim]  {msg}")

    storage_backend.init_db()

    # client.py is imported before the models are parsed: it is where a project
    # registers its own model types, and an unregistered type looks like a typo.
    if models_from_dict is None:
        from promptmap.llm import try_load_client_module

        try_load_client_module(map_dir)

    raw = models_from_dict.models if isinstance(models_from_dict, PromptModelsDict) else models_from_dict
    all_models = build_models_from_dict(raw) if models_from_dict is not None else load_models(map_dir)

    if select:
        import networkx as nx
        dag = build_dag(all_models)
        to_run: set[str] = set(select)
        for name in select:
            to_run.update(nx.ancestors(dag, name))
        ordered = [all_models[name] for name in to_run if name in all_models]
    else:
        ordered = list(all_models.values())

    git_sha = None
    if models_from_dict is None:
        import subprocess
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_sha = None

    # Resolve LLM and RAG backends from user files if not explicitly provided
    if llm_call is None:
        if models_from_dict is not None:
            raise ValueError(
                "llm_call must be provided when using models_from_dict. "
                "Inline runs do not auto-load client.py from disk."
            )
        from promptmap.llm import resolve_llm_call
        llm_call = resolve_llm_call(map_dir)
    if rag_call is None and models_from_dict is None:
        from promptmap.rag import resolve_rag_call
        rag_call = resolve_rag_call(map_dir)
    if blob_store is None and models_from_dict is None:
        from promptmap.llm import resolve_blob_store
        blob_store = resolve_blob_store(map_dir)
    if cache_check is None and models_from_dict is None:
        from promptmap.llm import resolve_cache_check
        cache_check = resolve_cache_check(map_dir)

    # Global instruction: explicit argument wins, else global.prompt on disk.
    from promptmap.global_instruction import (
        coerce_global_instruction,
        resolve_global_instruction,
    )
    global_instruction = coerce_global_instruction(global_instruction)
    if global_instruction is None and models_from_dict is None:
        global_instruction = resolve_global_instruction(map_dir)

    # Load per-model validators from validation_dir (optional)
    validators = None
    if validation_dir is not None and models_from_dict is None:
        from promptmap.validator import load_validators
        validators = load_validators(validation_dir)

    run_id = storage_backend.create_run(
        model_count=len(ordered),
        git_sha=git_sha,
    )
    from promptmap.run_ids import record_run_ids

    record_run_ids(
        storage_backend,
        run_id,
        all_models.values(),
        ordered,
        tests_dir=tests_dir if models_from_dict is None else None,
        global_instruction=global_instruction,
        promptdata=promptdata,
        promptfiles=promptfiles,
    )

    total = len(ordered)

    if verbose:
        console.print(
            f"[dim]{ts()}[/dim]  Running with [bold cyan]promptmap[/bold cyan]={__version__}"
        )
        console.print(
            f"[dim]{ts()}[/dim]  Found [bold]{total}[/bold] prompt model{'s' if total != 1 else ''}"
            + (f" [dim](select: {sorted(select)})[/dim]" if select else "")
        )
        if git_sha:
            console.print(f"[dim]{ts()}[/dim]  git SHA [dim]{git_sha}[/dim]")
        console.print(f"[dim]{ts()}[/dim]")

    completed: list[ModelRunResult] = []
    model_start_times: dict[str, float] = {}

    def _log_model_line(idx: int, verb: str, name: str, badge: str) -> None:
        prefix = f"{idx} of {total} {verb} prompt model {name} "
        dots = "." * max(2, 50 - len(prefix))
        if verbose:
            console.print(f"[dim]{ts()}[/dim]  [bold]{idx} of {total}[/bold] {verb} prompt model [cyan]{name}[/cyan] {dots} {badge}")

    def on_start(name: str) -> None:
        idx = len(completed) + 1
        model_start_times[name] = time.monotonic()
        _log_model_line(idx, "START", name, "[[yellow]RUN[/yellow]]")
        if on_model_start:
            on_model_start(name)

    def on_done(result: ModelRunResult) -> None:
        completed.append(result)
        idx = len(completed)
        elapsed = time.monotonic() - model_start_times.get(result.model_name, 0)
        if result.status == "success":
            cached = " cached" if result.cached else ""
            _log_model_line(idx, "OK   ", result.model_name, f"[[green]OK{cached} in {elapsed:.2f}s[/green]]")
        elif result.status == "skipped":
            _log_model_line(idx, "SKIP ", result.model_name, "[[yellow]SKIP[/yellow]]")
        else:
            _log_model_line(idx, "ERR  ", result.model_name, "[[red]ERROR[/red]]")
            if verbose:
                console.print(f"           [dim]{result.error}[/dim]")
        if on_model_done:
            on_model_done(result)

    run_start = time.monotonic()

    results = await execute_run(
        run_id=run_id,
        ordered_models=ordered,
        storage_backend=storage_backend,
        llm_call=llm_call,
        rag_call=rag_call,
        on_model_start=on_start,
        on_model_done=on_done,
        promptdata=promptdata,
        promptfiles=promptfiles,
        validators=validators or None,
        global_instruction=global_instruction,
        blob_store=blob_store,
        secrets=secrets,
        cache_check=cache_check,
    )

    errors = sum(1 for r in results if r.status == "error")
    skipped = sum(1 for r in results if r.status == "skipped")
    successes = sum(1 for r in results if r.status == "success")
    final_status = "success" if errors == 0 else ("partial" if successes > 0 else "error")
    storage_backend.finish_run(run_id, final_status)
    total_elapsed = time.monotonic() - run_start

    if verbose:
        console.print(f"[dim]{ts()}[/dim]")
        console.print(
            f"[dim]{ts()}[/dim]  Finished running {total} prompt model{'s' if total != 1 else ''} "
            f"in [bold]{total_elapsed:.2f}s[/bold]. "
            + ("[green]Completed successfully[/green]" if errors == 0 else "[red]Completed with errors[/red]")
        )
        pass_str  = f"[green]PASS={successes}[/green]"
        error_str = f"[red]ERROR={errors}[/red]"
        skip_str  = f"[yellow]SKIP={skipped}[/yellow]"
        total_str = f"TOTAL={total}"
        console.print(f"[dim]{ts()}[/dim]  {pass_str} {error_str} {skip_str} {total_str}")

    def _value(r):
        if r.status == "skipped":
            return ModelStatus.SKIPPED
        if r.status == "error":
            return ModelError(r.error or "error")
        from promptmap.files import contains_files

        if contains_files(r.value) or isinstance(r.value, str):
            return r.value
        return r.llm_output

    return {r.model_name: _value(r) for r in results}


def run(
    map_dir: str = "map",
    models_from_dict: "PromptModelsDict | dict[str, str] | None" = None,
    select: "list[str] | None" = None,
    llm_call: "Callable[[str], str | Awaitable[str]] | None" = None,
    rag_call: "Callable[..., list] | None" = None,
    verbose: bool = True,
    promptdata: "dict | None" = None,
    promptfiles: "dict[str, PromptFile] | None" = None,
    validation_dir: "str | None" = "validation",
    tests_dir: "str | None" = "tests",
    storage_backend: "StorageBackend | None" = None,
    global_instruction: "str | Callable[[], str] | None" = None,
    blob_store: "BlobStore | None" = None,
    on_model_start: "Callable[[str], None] | None" = None,
    on_model_done: "Callable[[ModelRunResult], None] | None" = None,
    secrets: "Mapping[str, str] | None" = None,
    cache_check: "Callable[..., bool | Awaitable[bool]] | None" = None,
):
    """Run prompt models synchronously.

    Wraps :func:`async_run` in ``asyncio.run()``; same arguments, same dict
    return value.  Call ``async_run`` directly from inside an event loop.
    """
    import asyncio

    return asyncio.run(async_run(
        map_dir=map_dir,
        models_from_dict=models_from_dict,
        select=select,
        llm_call=llm_call,
        rag_call=rag_call,
        verbose=verbose,
        promptdata=promptdata,
        promptfiles=promptfiles,
        validation_dir=validation_dir,
        tests_dir=tests_dir,
        storage_backend=storage_backend,
        global_instruction=global_instruction,
        blob_store=blob_store,
        on_model_start=on_model_start,
        on_model_done=on_model_done,
        secrets=secrets,
        cache_check=cache_check,
    ))
