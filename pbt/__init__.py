"""prompt-build-tool (pbt) — dbt-inspired LLM prompt orchestration."""

from __future__ import annotations

from enum import Enum
from typing import Callable

__version__ = "0.1.0"


class ModelStatus(Enum):
    """Returned in the pbt.run() dict for models that produced no LLM output."""
    SKIPPED = "skipped"          # upstream dependency failed → model was not run
    PROMPT_SKIPPED = "prompt_skipped"  # prompt rendered to "SKIP THIS MODEL"
    ERROR = "error"              # LLM call or render raised an exception


def run(
    models_dir: str = "models",
    select: list[str] | None = None,
    llm_call: Callable[[str], str] | None = None,
    rag_call: Callable[..., list] | None = None,
    verbose: bool = True,
):
    """
    Execute prompt models as a Python library call.

    Parameters
    ----------
    models_dir:
        Path to the directory containing *.prompt files.
    select:
        Optional list of model names to run. Upstream outputs are loaded
        from the most recent matching run in the DB.
    llm_call:
        Optional function ``(prompt: str) -> str`` to use as the LLM backend.
        Falls back to ``models/client.py`` then the built-in Gemini client.
    rag_call:
        Optional function ``(*args) -> list | str`` to back
        ``return_list_RAG_results()`` in templates.
        Falls back to ``models/rag.py::do_RAG`` if present.
    verbose:
        Print a dbt-style progress log to stdout (default: True).

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
    from pbt.executor import execute_run, ModelRunResult
    from pbt.parser import _SKIP_OUTPUT
    from pbt.graph import (
        load_models,
        execution_order,
        build_dag,
        compute_dag_hash,
    )

    console = Console(highlight=False, soft_wrap=True)

    def ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def log(msg: str) -> None:
        if verbose:
            console.print(f"[dim]{ts()}[/dim]  {msg}")

    db.init_db()

    all_models = load_models(models_dir)
    ordered = execution_order(all_models)
    dag_hash = compute_dag_hash(all_models)

    preloaded_outputs: dict[str, str] = {}

    if select:
        selected_set = set(select)
        dag = build_dag(all_models)

        upstream_needed: set[str] = set()
        for name in selected_set:
            upstream_needed.update(nx.ancestors(dag, name))
        upstream_needed -= selected_set

        if upstream_needed:
            prev_run = db.get_latest_run_with_dag_hash(dag_hash)
            if prev_run is None:
                raise RuntimeError(
                    f"select= requires a previous run with DAG hash '{dag_hash}'. "
                    "Call pbt.run() without select first."
                )
            preloaded_outputs = db.get_model_outputs_from_run(
                prev_run["run_id"], list(upstream_needed)
            )

        ordered = [m for m in ordered if m.name in selected_set]

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = None

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
            + (f", [dim]{len(preloaded_outputs)} upstream reused[/dim]" if preloaded_outputs else "")
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
        models_dir=models_dir,
        preloaded_outputs=preloaded_outputs,
        llm_call=llm_call,
        rag_call=rag_call,
        on_model_start=on_start,
        on_model_done=on_done,
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
        if r.llm_output == _SKIP_OUTPUT:
            return ModelStatus.PROMPT_SKIPPED
        return r.llm_output

    return {r.model_name: _value(r) for r in results}
