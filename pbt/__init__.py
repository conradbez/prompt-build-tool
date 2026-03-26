"""prompt-build-tool (pbt) — dbt-inspired LLM prompt orchestration."""

from __future__ import annotations

import re
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Awaitable, Callable

from pbt.storage.base import StorageBackend
from pbt.types import PromptFile, PromptModelsDict


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
        return package_version("prompt-build-tool")
    except PackageNotFoundError:
        return _version_from_pyproject()


__version__ = _detect_version()


class ModelStatus(Enum):
    """Returned in the pbt.run() dict for models that produced no LLM output."""
    SKIPPED = "skipped"          # upstream dependency failed → model was not run
    ERROR = "error"              # LLM call or render raised an exception (legacy)


class ModelError:
    """Returned by pbt.run() when a model fails; carries the exception message."""
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return f"error: {self.message}"


async def async_run(
    models_dir: str = "models",
    models_from_dict: PromptModelsDict | dict[str, str] | None = None,
    select: list[str] | None = None,
    llm_call: Callable[[str], str | Awaitable[str]] | None = None,
    rag_call: Callable[..., list] | None = None,
    verbose: bool = True,
    promptdata: dict | None = None,
    promptfiles: dict[str, PromptFile] | None = None,
    validation_dir: str | None = "validation",
    storage_backend: StorageBackend | None = None,
):
    """
    Execute prompt models as a Python library call.

    Parameters
    ----------
    models_dir:
        Path to the directory containing *.prompt files.
    select:
        Optional list of model names to run. All upstream dependencies are
        also executed fresh; the prompt cache makes unchanged nodes instant.
    llm_call:
        Optional function ``(prompt: str) -> str`` to use as the LLM backend.
        Falls back to ``models/client.py`` then the built-in Gemini client.
    rag_call:
        Optional function ``(*args) -> list | str`` to back
        ``return_list_RAG_results()`` in templates.
        Falls back to ``models/rag.py::do_RAG`` if present.
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


    Returns
    -------
    List of ``ModelRunResult`` (one per model executed).
    """
    import time
    from datetime import datetime

    from pbt.executor.executor import execute_run, ModelRunResult
    from pbt.executor.graph import (
        load_models,
        build_models_from_dict,
        build_dag,
    )
    if storage_backend is None:
        from pbt.storage.sqlite import SQLiteStorageBackend
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

    raw = models_from_dict.models if isinstance(models_from_dict, PromptModelsDict) else models_from_dict
    all_models = build_models_from_dict(raw) if models_from_dict is not None else load_models(models_dir)

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
        from pbt.llm import resolve_llm_call
        llm_call = resolve_llm_call(models_dir)
    if rag_call is None and models_from_dict is None:
        from pbt.rag import resolve_rag_call
        rag_call = resolve_rag_call(models_dir)

    # Load per-model validators from validation_dir (optional)
    validators = None
    if validation_dir is not None and models_from_dict is None:
        from pbt.validator import load_validators
        validators = load_validators(validation_dir)

    run_id = storage_backend.create_run(
        model_count=len(ordered),
        git_sha=git_sha,
    )

    total = len(ordered)

    if verbose:
        console.print(
            f"[dim]{ts()}[/dim]  Running with [bold cyan]pbt[/bold cyan]={__version__}"
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
        return r.llm_output

    return {r.model_name: _value(r) for r in results}


def run(
    models_dir: str = "models",
    models_from_dict: "PromptModelsDict | dict[str, str] | None" = None,
    select: "list[str] | None" = None,
    llm_call: "Callable[[str], str | Awaitable[str]] | None" = None,
    rag_call: "Callable[..., list] | None" = None,
    verbose: bool = True,
    promptdata: "dict | None" = None,
    promptfiles: "dict[str, PromptFile] | None" = None,
    validation_dir: "str | None" = "validation",
    storage_backend: "StorageBackend | None" = None,
):
    """Run prompt models synchronously."""
    import asyncio

    return asyncio.run(async_run(
        models_dir=models_dir,
        models_from_dict=models_from_dict,
        select=select,
        llm_call=llm_call,
        rag_call=rag_call,
        verbose=verbose,
        promptdata=promptdata,
        promptfiles=promptfiles,
        validation_dir=validation_dir,
        storage_backend=storage_backend,
    ))
