"""
pbt.cli.pretty_print
--------------------
Rich-based display helpers for the pbt CLI.

All console instances, table builders, and progress-callback factories live
here so that the main CLI module stays focused on orchestration logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table
from rich import box

from pbt.executor.executor import ModelRunResult
from pbt.tester import TestResult

# ---------------------------------------------------------------------------
# Shared consoles
# ---------------------------------------------------------------------------

console = Console()
err_console = Console(stderr=True, style="bold red")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "—"
    return str(ts)[:19].replace("T", " ")


# ---------------------------------------------------------------------------
# pbt run — header, progress callbacks, summary
# ---------------------------------------------------------------------------

def print_run_header(
    c: Console,
    run_id: str,
    dag_hash: str,
    ordered: list,
    select: tuple[str, ...],
    git_sha: str | None,
) -> None:
    c.rule("[bold cyan]pbt run[/bold cyan]")
    c.print(f"  Run ID   : [dim]{run_id}[/dim]")
    c.print(f"  DAG hash : [dim]{dag_hash}[/dim]")
    c.print(f"  Models   : {len(ordered)}", end="")
    if select:
        c.print(f"  [dim](select: {sorted(select)})[/dim]")
    else:
        c.print()
    if git_sha:
        c.print(f"  Git SHA  : [dim]{git_sha}[/dim]")
    c.print()


def make_run_callbacks(
    c: Console,
    results: list[ModelRunResult],
    total: int,
) -> tuple[Callable[[str], None], Callable[[ModelRunResult], None]]:
    """Return (on_start, on_done) callbacks for execute_run."""

    def on_start(name: str) -> None:
        idx = len(results) + 1
        c.print(f"  [{idx}/{total}] [bold]{name}[/bold] … ", end="")

    def on_done(result: ModelRunResult) -> None:
        results.append(result)
        if result.status == "success":
            c.print(f"[green]OK[/green] [dim]({result.execution_ms} ms)[/dim]")
        elif result.status == "skipped":
            c.print("[yellow]SKIPPED[/yellow]")
        else:
            c.print("[red]ERROR[/red]")
            c.print(f"    [dim]{result.error}[/dim]")

    return on_start, on_done


def print_run_summary(
    c: Console,
    all_results: list[ModelRunResult],
    outputs_dir: Path,
    written: list[str],
    run_id: str,
    dag_hash: str,
) -> None:
    successes = sum(1 for r in all_results if r.status == "success")
    errors    = sum(1 for r in all_results if r.status == "error")
    skipped   = sum(1 for r in all_results if r.status == "skipped")

    c.print()
    c.rule()

    summary = Table(box=box.SIMPLE, show_header=False)
    summary.add_row("Done    :", f"[green]{successes}[/green] succeeded")
    if errors:
        summary.add_row("        :", f"[red]{errors}[/red] errored")
    if skipped:
        summary.add_row("        :", f"[yellow]{skipped}[/yellow] skipped")
    if written:
        summary.add_row("Outputs :", f"[dim]{outputs_dir}/[/dim]  {', '.join(written)}")
    summary.add_row("Run ID  :", f"[dim]{run_id}[/dim]")
    summary.add_row("DAG hash:", f"[dim]{dag_hash}[/dim]")

    from pbt import db  # local import to avoid circular at module level
    summary.add_row("DB      :", f"[dim]{db.db_path()}[/dim]")
    c.print(summary)


# ---------------------------------------------------------------------------
# pbt test — header, progress callbacks, summary
# ---------------------------------------------------------------------------

def print_test_header(
    c: Console,
    tests_dir: str,
    tests: dict,
    target_run: dict,
    dag_hash: str,
) -> None:
    c.rule("[bold cyan]pbt test[/bold cyan]")
    c.print(f"  Tests dir  : [dim]{tests_dir}[/dim]")
    c.print(f"  Tests      : {len(tests)}")
    c.print(f"  Using run  : [dim]{target_run['run_id']}[/dim]  ({target_run['run_date']})")
    c.print(f"  DAG hash   : [dim]{dag_hash}[/dim]")
    c.print()


def make_test_callbacks(
    c: Console,
    test_results: list[TestResult],
    total: int,
) -> tuple[Callable[[str], None], Callable[[TestResult], None]]:
    """Return (on_start, on_done) callbacks for execute_tests."""

    def on_start(name: str) -> None:
        idx = len(test_results) + 1
        c.print(f"  [{idx}/{total}] [bold]{name}[/bold] … ", end="")

    def on_done(result: TestResult) -> None:
        test_results.append(result)
        if result.status == "pass":
            c.print(f"[green]PASS[/green] [dim]({result.execution_ms} ms)[/dim]")
        elif result.status == "fail":
            c.print("[red]FAIL[/red]")
            c.print(f"    LLM returned: [dim]{result.llm_output!r}[/dim]")
        else:
            c.print("[red]ERROR[/red]")
            c.print(f"    [dim]{result.error}[/dim]")

    return on_start, on_done


def print_test_summary(
    c: Console,
    test_results: list[TestResult],
    target_run: dict,
) -> None:
    passed  = sum(1 for r in test_results if r.status == "pass")
    failed  = sum(1 for r in test_results if r.status == "fail")
    errored = sum(1 for r in test_results if r.status == "error")

    c.print()
    c.rule()

    summary = Table(box=box.SIMPLE, show_header=False)
    summary.add_row("Passed  :", f"[green]{passed}[/green]")
    if failed:
        summary.add_row("Failed  :", f"[red]{failed}[/red]")
    if errored:
        summary.add_row("Errors  :", f"[red]{errored}[/red]")
    summary.add_row("Run ID  :", f"[dim]{target_run['run_id']}[/dim]")
    c.print(summary)


# ---------------------------------------------------------------------------
# pbt ls — models table
# ---------------------------------------------------------------------------

def models_table(ordered: list, dag_hash: str) -> Table:
    table = Table(
        title=f"Prompt Models  [dim](DAG hash: {dag_hash})[/dim]",
        box=box.ROUNDED,
    )
    table.add_column("#", style="dim", justify="right")
    table.add_column("Model", style="bold cyan")
    table.add_column("Depends on")
    table.add_column("promptdata() used", style="dim")
    table.add_column("File", style="dim")

    for i, model in enumerate(ordered, 1):
        deps = ", ".join(model.depends_on) if model.depends_on else "[dim]—[/dim]"
        promptdata_str = ", ".join(model.promptdata_used) if model.promptdata_used else "[dim]—[/dim]"
        table.add_row(str(i), model.name, deps, promptdata_str, str(model.path))

    return table


# ---------------------------------------------------------------------------
# pbt show-runs — runs table
# ---------------------------------------------------------------------------

_STATUS_STYLES = {
    "success": "green",
    "error": "red",
    "partial": "yellow",
    "running": "cyan",
}


def runs_table(rows: list) -> Table:
    table = Table(title="Recent Runs", box=box.ROUNDED)
    table.add_column("Run ID", style="dim", no_wrap=True)
    table.add_column("Date")
    table.add_column("Status")
    table.add_column("Models", justify="right")
    table.add_column("DAG hash", style="dim")
    table.add_column("Created at")
    table.add_column("Completed at")

    for row in rows:
        style = _STATUS_STYLES.get(row["status"], "")
        table.add_row(
            row["run_id"],
            row["run_date"] or "—",
            f"[{style}]{row['status']}[/{style}]",
            str(row["model_count"]),
            row["dag_hash"] or "—",
            _fmt_ts(row["created_at"]),
            _fmt_ts(row["completed_at"]) if row["completed_at"] else "—",
        )

    return table
