"""
pbt — prompt-build-tool CLI

Commands
--------
pbt run          Execute all prompt models (or a subset via --select).
pbt ls           List discovered models and their dependencies.
pbt show-runs    Show recent run history from the SQLite store.
pbt show-result  Print the stored output for a specific model + run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich import box

from pbt import db
from pbt.graph import load_models, execution_order, build_dag, CyclicDependencyError, UnknownModelError
from pbt.executor import execute_run, ModelRunResult

console = Console()
err_console = Console(stderr=True, style="bold red")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option()
def main() -> None:
    """prompt-build-tool (pbt) — dbt-inspired LLM prompt orchestration."""


# ---------------------------------------------------------------------------
# pbt run
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--models-dir",
    default="models",
    show_default=True,
    help="Directory containing *.prompt files.",
)
@click.option(
    "--select", "-s",
    multiple=True,
    metavar="MODEL",
    help="Run only these models (and their dependencies). Repeatable.",
)
@click.option(
    "--no-color",
    is_flag=True,
    default=False,
    help="Disable rich color output.",
)
def run(models_dir: str, select: tuple[str, ...], no_color: bool) -> None:
    """Execute all prompt models in dependency order."""
    c = Console(highlight=not no_color)

    db.init_db()

    # ------------------------------------------------------------------
    # Discover & validate models
    # ------------------------------------------------------------------
    try:
        all_models = load_models(models_dir)
    except FileNotFoundError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    try:
        ordered = execution_order(all_models)
    except (CyclicDependencyError, UnknownModelError) as exc:
        err_console.print(f"[red]Dependency error:[/red] {exc}")
        sys.exit(1)

    # Optional model selection (include dependencies automatically)
    if select:
        import networkx as nx
        dag = build_dag(all_models)
        selected_set: set[str] = set(select)
        for name in select:
            if name not in all_models:
                err_console.print(f"[red]Unknown model:[/red] '{name}'")
                sys.exit(1)
            # Include all ancestors (upstream dependencies)
            selected_set.update(nx.ancestors(dag, name))
        ordered = [m for m in ordered if m.name in selected_set]

    # ------------------------------------------------------------------
    # Print run header
    # ------------------------------------------------------------------
    git_sha = _git_sha()
    run_id = db.create_run(model_count=len(ordered), git_sha=git_sha)

    c.rule("[bold cyan]pbt run[/bold cyan]")
    c.print(f"  Run ID  : [dim]{run_id}[/dim]")
    c.print(f"  Models  : {len(ordered)}")
    if git_sha:
        c.print(f"  Git SHA : [dim]{git_sha}[/dim]")
    c.print()

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    results: list[ModelRunResult] = []
    total = len(ordered)

    def on_start(name: str) -> None:
        idx = sum(1 for r in results if r.status != "pending") + 1
        c.print(f"  [{idx}/{total}] [bold]{name}[/bold] … ", end="")

    def on_done(result: ModelRunResult) -> None:
        results.append(result)
        if result.status == "success":
            ms = result.execution_ms
            c.print(f"[green]OK[/green] [dim]({ms} ms)[/dim]")
        elif result.status == "skipped":
            c.print("[yellow]SKIPPED[/yellow]")
        else:
            c.print(f"[red]ERROR[/red]")
            c.print(f"    [dim]{result.error}[/dim]")

    try:
        all_results = execute_run(
            run_id=run_id,
            ordered_models=ordered,
            on_model_start=on_start,
            on_model_done=on_done,
        )
    except EnvironmentError as exc:
        err_console.print(f"\n[red]Configuration error:[/red] {exc}")
        db.finish_run(run_id, "error")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    successes = sum(1 for r in all_results if r.status == "success")
    errors    = sum(1 for r in all_results if r.status == "error")
    skipped   = sum(1 for r in all_results if r.status == "skipped")

    final_status = "success" if errors == 0 else ("partial" if successes > 0 else "error")
    db.finish_run(run_id, final_status)

    c.print()
    c.rule()

    summary = Table(box=box.SIMPLE, show_header=False)
    summary.add_row("Done  :", f"[green]{successes}[/green] succeeded")
    if errors:
        summary.add_row("      :", f"[red]{errors}[/red] errored")
    if skipped:
        summary.add_row("      :", f"[yellow]{skipped}[/yellow] skipped")
    summary.add_row("Run ID:", f"[dim]{run_id}[/dim]")
    summary.add_row("DB    :", f"[dim]{db.db_path()}[/dim]")
    c.print(summary)

    if errors:
        sys.exit(1)


# ---------------------------------------------------------------------------
# pbt ls
# ---------------------------------------------------------------------------

@main.command("ls")
@click.option("--models-dir", default="models", show_default=True)
def list_models(models_dir: str) -> None:
    """List all discovered models and their dependencies."""
    try:
        models = load_models(models_dir)
        ordered = execution_order(models)
    except (FileNotFoundError, CyclicDependencyError, UnknownModelError) as exc:
        err_console.print(str(exc))
        sys.exit(1)

    table = Table(title="Prompt Models", box=box.ROUNDED)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Model", style="bold cyan")
    table.add_column("Depends on")
    table.add_column("File", style="dim")

    for i, model in enumerate(ordered, 1):
        deps = ", ".join(model.depends_on) if model.depends_on else "[dim]—[/dim]"
        table.add_row(str(i), model.name, deps, str(model.path))

    console.print(table)


# ---------------------------------------------------------------------------
# pbt show-runs
# ---------------------------------------------------------------------------

@main.command("show-runs")
@click.option("--limit", default=10, show_default=True, help="Number of runs to show.")
def show_runs(limit: int) -> None:
    """Show recent run history."""
    db.init_db()
    rows = db.get_latest_runs(limit)

    if not rows:
        console.print("[dim]No runs recorded yet. Run `pbt run` first.[/dim]")
        return

    table = Table(title="Recent Runs", box=box.ROUNDED)
    table.add_column("Run ID", style="dim", no_wrap=True)
    table.add_column("Status")
    table.add_column("Models", justify="right")
    table.add_column("Created at")
    table.add_column("Completed at")

    status_styles = {"success": "green", "error": "red", "partial": "yellow", "running": "cyan"}

    for row in rows:
        style = status_styles.get(row["status"], "")
        table.add_row(
            row["run_id"],
            f"[{style}]{row['status']}[/{style}]",
            str(row["model_count"]),
            _fmt_ts(row["created_at"]),
            _fmt_ts(row["completed_at"]) if row["completed_at"] else "—",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# pbt show-result
# ---------------------------------------------------------------------------

@main.command("show-result")
@click.argument("model_name")
@click.option("--run-id", default=None, help="Specific run ID (defaults to latest).")
@click.option(
    "--show",
    type=click.Choice(["output", "prompt", "all"]),
    default="output",
    show_default=True,
)
def show_result(model_name: str, run_id: str | None, show: str) -> None:
    """Print stored output for MODEL_NAME."""
    db.init_db()

    with db.get_conn() as conn:
        if run_id:
            row = conn.execute(
                "SELECT * FROM model_results WHERE run_id=? AND model_name=?",
                (run_id, model_name),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT mr.* FROM model_results mr
                   JOIN runs r ON r.run_id = mr.run_id
                   WHERE mr.model_name = ?
                   ORDER BY r.created_at DESC LIMIT 1""",
                (model_name,),
            ).fetchone()

    if not row:
        err_console.print(f"No result found for model '{model_name}'.")
        sys.exit(1)

    console.rule(f"[bold]{model_name}[/bold] — run [dim]{row['run_id']}[/dim]")
    console.print(f"Status      : {row['status']}")
    console.print(f"Execution   : {row['execution_ms']} ms")

    if show in ("prompt", "all"):
        console.rule("[dim]Rendered prompt[/dim]")
        console.print(row["prompt_rendered"] or "")

    if show in ("output", "all"):
        console.rule("[dim]LLM output[/dim]")
        console.print(row["llm_output"] or "")

    if row["error"]:
        console.rule("[red]Error[/red]")
        console.print(row["error"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "—"
    # SQLite stores ISO strings; just return the readable part
    return str(ts)[:19].replace("T", " ")
