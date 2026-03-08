"""
pbt — prompt-build-tool CLI

Commands
--------
pbt run          Execute all prompt models (or a subset via --select).
pbt test         Run test prompts from the tests/ directory.
pbt ls           List discovered models and their dependencies.
pbt show-runs    Show recent run history from the SQLite store.
pbt show-result  Print the stored output for a specific model + run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
import networkx as nx
from rich.console import Console
from rich.table import Table
from rich import box

from pbt import db
from pbt.graph import (
    load_models,
    execution_order,
    build_dag,
    compute_dag_hash,
    CyclicDependencyError,
    UnknownModelError,
)
from pbt.executor import execute_run, ModelRunResult
from pbt.llm import resolve_llm_call
from pbt.rag import resolve_rag_call
from pbt.tester import load_tests, execute_tests, TestResult
from pbt.docs import generate_docs
from pbt.validator import load_validators

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
    help=(
        "Run only these models, reusing upstream outputs from the latest "
        "matching run. Repeatable: -s tweet -s haiku"
    ),
)
@click.option(
    "--no-color",
    is_flag=True,
    default=False,
    help="Disable rich color output.",
)
@click.option(
    "--var",
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Inject a variable into every Jinja2 template. "
        "Repeatable: --var country=USA --var max_tokens=200"
    ),
)
@click.option(
    "--validation-dir",
    default="validation",
    show_default=True,
    help="Directory containing per-model validation Python files.",
)
def run(models_dir: str, select: tuple[str, ...], no_color: bool, var: tuple[str, ...], validation_dir: str) -> None:
    """Execute all prompt models in dependency order."""
    c = Console(highlight=not no_color)

    # Parse --var KEY=VALUE pairs into a dict
    extra_vars: dict[str, str] = {}
    for v in var:
        if "=" not in v:
            err_console.print(f"[red]Error:[/red] --var must be KEY=VALUE, got: {v!r}")
            sys.exit(1)
        k, _, val = v.partition("=")
        extra_vars[k] = val

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

    dag_hash = compute_dag_hash(all_models)

    # ------------------------------------------------------------------
    # --select: only run chosen models, load upstream outputs from DB
    # ------------------------------------------------------------------
    preloaded_outputs: dict[str, str] = {}

    if select:
        # Validate names
        for name in select:
            if name not in all_models:
                err_console.print(f"[red]Unknown model:[/red] '{name}'")
                sys.exit(1)

        selected_set = set(select)
        dag = build_dag(all_models)

        # Determine which upstream models we need outputs for but won't run
        upstream_needed: set[str] = set()
        for name in selected_set:
            upstream_needed.update(nx.ancestors(dag, name))
        upstream_needed -= selected_set

        if upstream_needed:
            prev_run = db.get_latest_run_with_dag_hash(dag_hash)
            if prev_run is None:
                err_console.print(
                    f"[red]Error:[/red] --select requires a previous run with the "
                    f"same DAG structure (hash [bold]{dag_hash}[/bold]).\n"
                    f"Run [bold]pbt run[/bold] without --select first."
                )
                sys.exit(1)

            preloaded_outputs = db.get_model_outputs_from_run(
                prev_run["run_id"], list(upstream_needed)
            )

            missing = upstream_needed - set(preloaded_outputs)
            if missing:
                err_console.print(
                    f"[red]Error:[/red] Previous run [dim]{prev_run['run_id']}[/dim] "
                    f"is missing successful outputs for: {sorted(missing)}\n"
                    f"Those models may have errored. Run [bold]pbt run[/bold] first."
                )
                sys.exit(1)

        # Only execute the explicitly selected models (in topological order)
        ordered = [m for m in ordered if m.name in selected_set]

    # ------------------------------------------------------------------
    # Print run header
    # ------------------------------------------------------------------
    git_sha = _git_sha()
    run_id = db.create_run(
        model_count=len(ordered),
        dag_hash=dag_hash,
        git_sha=git_sha,
    )

    c.rule("[bold cyan]pbt run[/bold cyan]")
    c.print(f"  Run ID   : [dim]{run_id}[/dim]")
    c.print(f"  DAG hash : [dim]{dag_hash}[/dim]")
    c.print(f"  Models   : {len(ordered)}", end="")
    if preloaded_outputs:
        c.print(
            f"  [dim](+{len(preloaded_outputs)} reused from previous run)[/dim]"
        )
    else:
        c.print()
    if git_sha:
        c.print(f"  Git SHA  : [dim]{git_sha}[/dim]")
    c.print()

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    results: list[ModelRunResult] = []
    total = len(ordered)

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

    # Discover user-provided client.py and rag.py from models_dir
    try:
        llm_call = resolve_llm_call(models_dir)
        rag_call = resolve_rag_call(models_dir)
    except Exception as exc:
        err_console.print(f"[red]Backend resolution error:[/red] {exc}")
        db.finish_run(run_id, "error")
        sys.exit(1)

    # Load per-model validators from validation_dir (optional)
    try:
        validators = load_validators(validation_dir)
    except AttributeError as exc:
        err_console.print(f"[red]Validation config error:[/red] {exc}")
        db.finish_run(run_id, "error")
        sys.exit(1)

    if validators:
        c.print(f"  Validators: {sorted(validators.keys())}")
        c.print()

    if extra_vars:
        c.print(f"  Vars     : {extra_vars}")
        c.print()

    try:
        all_results = execute_run(
            run_id=run_id,
            ordered_models=ordered,
            preloaded_outputs=preloaded_outputs,
            on_model_start=on_start,
            on_model_done=on_done,
            llm_call=llm_call,
            rag_call=rag_call,
            vars=extra_vars or None,
            validators=validators or None,
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
    summary.add_row("Done    :", f"[green]{successes}[/green] succeeded")
    if errors:
        summary.add_row("        :", f"[red]{errors}[/red] errored")
    if skipped:
        summary.add_row("        :", f"[yellow]{skipped}[/yellow] skipped")
    if preloaded_outputs:
        summary.add_row("Reused  :", f"[dim]{len(preloaded_outputs)} from previous run[/dim]")
    summary.add_row("Run ID  :", f"[dim]{run_id}[/dim]")
    summary.add_row("DAG hash:", f"[dim]{dag_hash}[/dim]")
    summary.add_row("DB      :", f"[dim]{db.db_path()}[/dim]")
    c.print(summary)

    if errors:
        sys.exit(1)


# ---------------------------------------------------------------------------
# pbt test
# ---------------------------------------------------------------------------

@main.command("test")
@click.option(
    "--models-dir",
    default="models",
    show_default=True,
    help="Directory containing *.prompt model files.",
)
@click.option(
    "--tests-dir",
    default="tests",
    show_default=True,
    help="Directory containing *.prompt test files.",
)
@click.option(
    "--run-id",
    default=None,
    help="Use outputs from this specific run (default: latest run with matching DAG hash).",
)
@click.option("--no-color", is_flag=True, default=False)
def test(models_dir: str, tests_dir: str, run_id: str | None, no_color: bool) -> None:
    """
    Run test prompts from the tests/ directory against the latest model outputs.

    Each test prompt has full Jinja2 context (ref() works as in models).
    A test passes when the LLM returns JSON containing {"results": "pass"}.
    """
    c = Console(highlight=not no_color)
    db.init_db()

    # ------------------------------------------------------------------
    # Discover tests
    # ------------------------------------------------------------------
    tests = load_tests(tests_dir)
    if not tests:
        c.print(
            f"[yellow]No test files found in '{tests_dir}'.[/yellow]\n"
            f"Create *.prompt files there to get started."
        )
        return

    # ------------------------------------------------------------------
    # Load models to compute the DAG hash
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

    dag_hash = compute_dag_hash(all_models)

    # ------------------------------------------------------------------
    # Find the run whose outputs we'll test against
    # ------------------------------------------------------------------
    if run_id:
        with db.get_conn() as conn:
            target_run = conn.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if not target_run:
            err_console.print(f"[red]Error:[/red] Run '{run_id}' not found.")
            sys.exit(1)
    else:
        target_run = db.get_latest_run_with_dag_hash(dag_hash)
        if target_run is None:
            err_console.print(
                f"[red]Error:[/red] No previous run found with DAG hash [bold]{dag_hash}[/bold].\n"
                f"Run [bold]pbt run[/bold] first, then [bold]pbt test[/bold]."
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # Load model outputs from that run
    # ------------------------------------------------------------------
    model_names = [m.name for m in ordered]
    model_outputs = db.get_model_outputs_from_run(target_run["run_id"], model_names)

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    c.rule("[bold cyan]pbt test[/bold cyan]")
    c.print(f"  Tests dir  : [dim]{tests_dir}[/dim]")
    c.print(f"  Tests      : {len(tests)}")
    c.print(f"  Using run  : [dim]{target_run['run_id']}[/dim]  ({target_run['run_date']})")
    c.print(f"  DAG hash   : [dim]{dag_hash}[/dim]")
    c.print()

    # ------------------------------------------------------------------
    # Execute tests
    # ------------------------------------------------------------------
    test_results: list[TestResult] = []
    total = len(tests)

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

    execute_tests(
        run_id=target_run["run_id"],
        tests=tests,
        model_outputs=model_outputs,
        on_test_start=on_start,
        on_test_done=on_done,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
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

    if failed or errored:
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

    dag_hash = compute_dag_hash(models)

    table = Table(
        title=f"Prompt Models  [dim](DAG hash: {dag_hash})[/dim]",
        box=box.ROUNDED,
    )
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
    table.add_column("Date")
    table.add_column("Status")
    table.add_column("Models", justify="right")
    table.add_column("DAG hash", style="dim")
    table.add_column("Created at")
    table.add_column("Completed at")

    status_styles = {
        "success": "green",
        "error": "red",
        "partial": "yellow",
        "running": "cyan",
    }

    for row in rows:
        style = status_styles.get(row["status"], "")
        table.add_row(
            row["run_id"],
            row["run_date"] or "—",
            f"[{style}]{row['status']}[/{style}]",
            str(row["model_count"]),
            row["dag_hash"] or "—",
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
# pbt docs
# ---------------------------------------------------------------------------

@main.command("docs")
@click.option(
    "--models-dir",
    default="models",
    show_default=True,
    help="Directory containing *.prompt files (for DAG diagram).",
)
@click.option(
    "--output",
    default=".pbt/docs/index.html",
    show_default=True,
    help="Path to write the generated HTML file.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    default=False,
    help="Open the generated file in the default browser.",
)
def docs(models_dir: str, output: str, open_browser: bool) -> None:
    """Generate a self-contained HTML report of all previous runs."""
    import webbrowser

    db.init_db()

    # Load all runs and their model results
    all_runs = db.get_latest_runs(limit=10_000)
    run_results: dict = {}
    for run in all_runs:
        run_results[run["run_id"]] = db.get_run_results(run["run_id"])

    # Try to load models for DAG diagram (optional)
    models = None
    try:
        models = load_models(models_dir)
    except (FileNotFoundError, Exception):
        pass

    output_path = Path(output)
    generate_docs(
        runs=list(all_runs),
        run_results=run_results,
        models=models,
        output_path=output_path,
    )

    console.print(f"[green]Docs generated:[/green] [bold]{output_path}[/bold]")

    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())


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
    return str(ts)[:19].replace("T", " ")
