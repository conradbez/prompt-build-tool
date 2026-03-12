"""prompt-build-tool (pbt) — dbt-inspired LLM prompt orchestration."""

from __future__ import annotations

from enum import Enum
from typing import Callable

from pbt.types import PromptFile, PromptModelsDict

__version__ = "0.1.0"


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


def run(
    models_dir: str = "models",
    models_from_dict: PromptModelsDict | dict[str, str] | None = None,
    select: list[str] | None = None,
    dag_id: str | None = None,
    llm_call: Callable[[str], str] | None = None,
    rag_call: Callable[..., list] | None = None,
    verbose: bool = True,
    promptdata: dict | None = None,
    promptfiles: dict[str, PromptFile] | None = None,
    validation_dir: str = "validation",
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
    dag_id:
        Optional DAG hash returned by a previous run. When provided, the DAG
        (model sources and config) is loaded from the database instead of
        reading *.prompt files from disk.
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
    import subprocess
    import time
    from datetime import datetime

    import networkx as nx
    from rich.console import Console

    from pbt import db
    from pbt.executor.executor import execute_run, ModelRunResult
    from pbt.llm import resolve_llm_call
    from pbt.rag import resolve_rag_call
    from pbt.validator import load_validators
    from pbt.executor.graph import (
        load_models,
        build_models_from_dict,
        execution_order,
        build_dag,
        compute_dag_hash,
        models_to_json,
        models_from_json,
    )

    console = Console(highlight=False, soft_wrap=True)

    def ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def log(msg: str) -> None:
        if verbose:
            console.print(f"[dim]{ts()}[/dim]  {msg}")

    db.init_db()

    if dag_id:
        dag_json = db.load_dag(dag_id)
        if dag_json is None:
            raise RuntimeError(
                f"DAG '{dag_id}' not found in database. "
                "Run pbt.run() without dag_id first to register it."
            )
        all_models = models_from_json(dag_json)
        dag_hash = dag_id
    elif models_from_dict is not None:
        raw = models_from_dict.root if isinstance(models_from_dict, PromptModelsDict) else models_from_dict
        all_models = build_models_from_dict(raw)
        dag_hash = compute_dag_hash(all_models)
        db.save_dag(dag_hash, models_to_json(all_models))
    else:
        all_models = load_models(models_dir)
        dag_hash = compute_dag_hash(all_models)
        db.save_dag(dag_hash, models_to_json(all_models))

    ordered = execution_order(all_models)

    if select:
        selected_set = set(select)
        dag = build_dag(all_models)

        # Run selected nodes AND all their ancestors fresh.
        # The prompt cache makes unchanged upstream nodes instant.
        to_run: set[str] = set(selected_set)
        for name in selected_set:
            to_run.update(nx.ancestors(dag, name))

        ordered = [m for m in ordered if m.name in to_run]

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = None

    # Resolve LLM and RAG backends from user files if not explicitly provided
    if llm_call is None:
        llm_call = resolve_llm_call(models_dir)
    if rag_call is None:
        rag_call = resolve_rag_call(models_dir)

    # Load per-model validators from validation_dir (optional)
    validators = load_validators(validation_dir)

    run_id = db.create_run(
        model_count=len(ordered),
        dag_hash=dag_hash,
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

    results = execute_run(
        run_id=run_id,
        ordered_models=ordered,
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
    db.finish_run(run_id, final_status)
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
            return ModelStatus.ERROR
        return r.llm_output

    return {r.model_name: _value(r) for r in results}
