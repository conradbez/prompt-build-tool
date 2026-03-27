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

import asyncio
import subprocess
import sys
from pathlib import Path

import click
import networkx as nx
from rich.console import Console

from pbt import db
from pbt.executor.graph import (
    load_models,
    build_dag,
    get_dag_promptdata,
    get_dag_promptfiles,
    CyclicDependencyError,
    UnknownModelError,
)
from pbt.executor.executor import execute_run
from pbt.llm import resolve_llm_call
from pbt.rag import resolve_rag_call
from pbt.tester import load_tests, execute_tests
from pbt.promptparams import load_promptparams, write_example
from pbt.docs import generate_docs
from pbt.validator import load_validators
from pbt.cli.vscode import is_running_in_vscode, setup_vscode_associations
from pbt.cli.type_hints import register_command as _register_type_hints, generate_stubs as _generate_stubs
from pbt.cli.init_files import register_command as _register_init
from pbt.cli import pretty_print
from pbt.cli.pretty_print import console, err_console


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
        "Run only these models and their upstream dependencies. "
        "Unchanged nodes are served instantly from the prompt cache. "
        "Repeatable: -s tweet -s haiku"
    ),
)
@click.option(
    "--no-color",
    is_flag=True,
    default=False,
    help="Disable rich color output.",
)
@click.option(
    "--promptdata",
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Inject a variable into every Jinja2 template via promptdata(). "
        "Repeatable: --promptdata country=USA --promptdata tone=formal"
    ),
)
@click.option(
    "--promptfile",
    "promptfiles",
    multiple=True,
    metavar="NAME=PATH",
    help=(
        "Provide a file by name for models that declare it in their config block. "
        "Repeatable: --promptfile doc=report.pdf --promptfile img=chart.png"
    ),
)
@click.option(
    "--validation-dir",
    default="validation",
    show_default=True,
    help="Directory containing per-model validation Python files.",
)
@click.option(
    "--clear-cache",
    is_flag=True,
    default=False,
    help="Clear the LLM prompt cache before running, forcing fresh calls for all models.",
)
def run(models_dir: str, select: tuple[str, ...], no_color: bool, promptdata: tuple[str, ...], promptfiles: tuple[str, ...], validation_dir: str, clear_cache: bool) -> None:
    """Execute all prompt models in dependency order."""
    c = Console(highlight=not no_color)

    if clear_cache:
        db.init_db()
        cleared = db.clear_cache()
        c.print(f"  [yellow]Cache cleared[/yellow] ({cleared} entr{'ies' if cleared != 1 else 'y'} invalidated)\n")

    # Parse --promptdata KEY=VALUE pairs into a dict
    promptdata_vars: dict[str, str] = {}
    for v in promptdata:
        if "=" not in v:
            err_console.print(f"[red]Error:[/red] --promptdata must be KEY=VALUE, got: {v!r}")
            sys.exit(1)
        k, _, val = v.partition("=")
        promptdata_vars[k] = val

    # Parse --promptfile NAME=PATH pairs into a dict of open file objects.
    promptfiles_dict: dict = {}
    for f in promptfiles:
        if "=" not in f:
            err_console.print(f"[red]Error:[/red] --promptfile must be NAME=PATH, got: {f!r}")
            sys.exit(1)
        k, _, val = f.partition("=")
        try:
            promptfiles_dict[k] = open(val, "rb")  # noqa: WPS515  – closed by llm_call consumer
        except OSError as exc:
            err_console.print(f"[red]Error:[/red] Cannot open promptfile '{val}': {exc}")
            sys.exit(1)

    db.init_db()

    # ------------------------------------------------------------------
    # Discover & validate models
    # ------------------------------------------------------------------
    try:
        all_models = load_models(models_dir)
    except FileNotFoundError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    # ------------------------------------------------------------------
    # --select: run chosen models AND their full upstream dependency chain.
    # ------------------------------------------------------------------
    if select:
        for name in select:
            if name not in all_models:
                err_console.print(f"[red]Unknown model:[/red] '{name}'")
                sys.exit(1)
        try:
            dag = build_dag(all_models)
        except (CyclicDependencyError, UnknownModelError) as exc:
            err_console.print(f"[red]Dependency error:[/red] {exc}")
            sys.exit(1)
        to_run: set[str] = set(select)
        for name in select:
            to_run.update(nx.ancestors(dag, name))
        ordered = [all_models[name] for name in to_run]
    else:
        try:
            build_dag(all_models)  # validate only
        except (CyclicDependencyError, UnknownModelError) as exc:
            err_console.print(f"[red]Dependency error:[/red] {exc}")
            sys.exit(1)
        ordered = list(all_models.values())

    # ------------------------------------------------------------------
    # Print run header
    # ------------------------------------------------------------------
    git_sha = _git_sha()
    run_id = db.create_run(
        model_count=len(ordered),
        git_sha=git_sha,
    )

    pretty_print.print_run_header(c, run_id, ordered, select, git_sha)

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    results: list = []
    on_start, on_done = pretty_print.make_run_callbacks(c, results, total=len(ordered))

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

    orphan_validators = [v for v in validators if v not in all_models]
    if orphan_validators:
        err_console.print(
            f"[red]Error:[/red] validation files have no matching model: {orphan_validators}\n"
            f"Rename or remove them to match a .prompt file."
        )
        db.finish_run(run_id, "error")
        sys.exit(1)

    if validators:
        c.print(f"  Validators: {sorted(validators.keys())}")
        c.print()

    # Warn about promptdata() vars used in templates but not provided
    dag_promptdata = get_dag_promptdata(all_models)
    missing_promptdata = [v for v in dag_promptdata if v not in promptdata_vars]
    if promptdata_vars:
        c.print(f"  promptdata() set : {list(promptdata_vars.keys())}")
    if missing_promptdata:
        c.print(f"  [yellow]Warning: promptdata() vars not provided: {missing_promptdata}[/yellow]")
    if dag_promptdata or promptdata_vars:
        c.print()

    try:
        all_results = asyncio.run(execute_run(
            run_id=run_id,
            ordered_models=ordered,
            storage_backend=db,
            on_model_start=on_start,
            on_model_done=on_done,
            llm_call=llm_call,
            rag_call=rag_call,
            promptdata=promptdata_vars or None,
            promptfiles=promptfiles_dict or None,
            validators=validators or None,
        ))
    except EnvironmentError as exc:
        err_console.print(f"\n[red]Configuration error:[/red] {exc}")
        db.finish_run(run_id, "error")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    errors = sum(1 for r in all_results if r.status == "error")
    successes = sum(1 for r in all_results if r.status == "success")

    final_status = "success" if errors == 0 else ("partial" if successes > 0 else "error")
    db.finish_run(run_id, final_status)

    # Write outputs/ directory — one file per successful model.
    # Extension comes from {{ config(output_extension="html") }}; defaults to "md".
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    written: list[str] = []
    for result in all_results:
        if result.status == "success" and result.llm_output:
            ext = all_models[result.model_name].config.get("output_extension", "md")
            out_file = outputs_dir / f"{result.model_name}.{ext}"
            out_file.write_text(result.llm_output, encoding="utf-8")
            written.append(result.model_name)

    pretty_print.print_run_summary(c, all_results, outputs_dir, written, run_id)

    if errors:
        sys.exit(1)

    try:
        _generate_stubs(validation_dir)
    except Exception:
        pass  # type-hint generation is best-effort; never block a run


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
    help="Use outputs from this specific run (default: latest run). Ignored when --promptparams-file rows are found.",
)
@click.option("--no-color", is_flag=True, default=False)
@click.option(
    "--promptparams-file",
    default="promptparams.csv",
    show_default=True,
    help=(
        "CSV file with columns for promptdata / promptfile parameters. "
        "When rows are present, pbt run is executed for each row and tests "
        "are reported per row. Column names: promptdata.<key> or promptfile.<name>. "
        "Ignored when the file does not exist."
    ),
)
@click.option(
    "--check-latest",
    is_flag=True,
    default=False,
    help="Skip promptparams.csv and test against the latest stored run instead.",
)
def test(
    models_dir: str,
    tests_dir: str,
    run_id: str | None,
    no_color: bool,
    promptparams_file: str,
    check_latest: bool,
) -> None:
    """
    Run test prompts from the tests/ directory against model outputs.

    Each test prompt has full Jinja2 context (ref() works as in models).
    A test passes when the LLM returns JSON containing {"results": "pass"}.

    When a promptparams.csv file is present, pbt run is executed once per row
    and tests are reported for each row individually.
    Without promptparams, tests run against the latest (or specified) run.
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
    # Load models
    # ------------------------------------------------------------------
    try:
        all_models = load_models(models_dir)
    except FileNotFoundError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve LLM backend (needed for both modes)
    # ------------------------------------------------------------------
    try:
        llm_call = resolve_llm_call(models_dir)
        rag_call = resolve_rag_call(models_dir)
    except Exception as exc:
        err_console.print(f"[red]Backend resolution error:[/red] {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Write promptparams.csv.example — column template for this DAG
    # ------------------------------------------------------------------
    from pbt.executor.parser_initial import detect_used_promptdata

    dag_promptdata = get_dag_promptdata(all_models)
    for src in tests.values():
        for key in detect_used_promptdata(src):
            if key not in dag_promptdata:
                dag_promptdata.append(key)
    dag_promptfiles = get_dag_promptfiles(all_models)

    example_path = Path(tests_dir) / "promptparams.csv.example"
    try:
        write_example(example_path, dag_promptdata, dag_promptfiles)
        if dag_promptdata or dag_promptfiles:
            c.print(f"  [dim]promptparams.csv.example written → {example_path}[/dim]")
            c.print()
    except Exception:  # noqa: BLE001
        pass

    # ------------------------------------------------------------------
    # Load promptparams rows (optional; skipped when --check-latest)
    # ------------------------------------------------------------------
    promptparams_rows = [] if check_latest else load_promptparams(promptparams_file)

    if promptparams_rows:
        # --------------------------------------------------------------
        # Per-row mode: run models then test for each CSV row
        # --------------------------------------------------------------
        from pbt.executor.executor import execute_run
        from pbt.promptparams import parse_promptparams_row

        c.print(
            f"  promptparams : [dim]{promptparams_file}[/dim] "
            f"({len(promptparams_rows)} row{'s' if len(promptparams_rows) != 1 else ''})"
        )
        c.print()

        try:
            ordered_models = list(build_dag(all_models))  # validates DAG
        except (CyclicDependencyError, UnknownModelError) as exc:
            err_console.print(f"[red]Dependency error:[/red] {exc}")
            sys.exit(1)
        ordered_models = list(all_models.values())

        git_sha = _git_sha()
        all_test_results: list = []

        for idx, row in enumerate(promptparams_rows, start=1):
            row_promptdata, row_promptfiles = parse_promptparams_row(row)
            row_label = ", ".join(f"{k}={v}" for k, v in row.items() if v)
            c.rule(f"[bold]Row {idx}[/bold]" + (f" — {row_label}" if row_label else ""))

            # Run models for this row
            row_run_id = db.create_run(model_count=len(ordered_models), git_sha=git_sha)
            model_run_results: list = []
            on_model_start, on_model_done = pretty_print.make_run_callbacks(c, model_run_results, total=len(ordered_models))
            run_results = asyncio.run(execute_run(
                run_id=row_run_id,
                ordered_models=ordered_models,
                storage_backend=db,
                on_model_start=on_model_start,
                on_model_done=on_model_done,
                llm_call=llm_call,
                rag_call=rag_call,
                promptdata=row_promptdata or None,
                promptfiles=row_promptfiles or None,
            ))
            run_errors = sum(1 for r in run_results if r.status == "error")
            db.finish_run(row_run_id, "success" if not run_errors else "partial")

            model_outputs = {
                r.model_name: r.llm_output
                for r in run_results
                if r.status == "success" and r.llm_output
            }

            # Run tests against this row's outputs
            row_test_results: list = []
            on_start, on_done = pretty_print.make_test_callbacks(c, row_test_results, total=len(tests))
            execute_tests(
                run_id=row_run_id,
                tests=tests,
                model_outputs=model_outputs,
                storage_backend=db,
                on_test_start=on_start,
                on_test_done=on_done,
                llm_call=llm_call,
            )
            all_test_results.extend(row_test_results)

            passed = sum(1 for r in row_test_results if r.status == "pass")
            failed_row = sum(1 for r in row_test_results if r.status in ("fail", "error"))
            status_color = "green" if not failed_row else "red"
            c.print(f"  [{status_color}]{passed}/{len(row_test_results)} passed[/{status_color}]")
            c.print()

        # Overall summary
        total_passed = sum(1 for r in all_test_results if r.status == "pass")
        total_failed = sum(1 for r in all_test_results if r.status in ("fail", "error"))
        c.rule("[bold]Overall[/bold]")
        overall_color = "green" if not total_failed else "red"
        c.print(f"  [{overall_color}]{total_passed}/{len(all_test_results)} passed across {len(promptparams_rows)} rows[/{overall_color}]")

        if total_failed:
            sys.exit(1)

    else:
        # --------------------------------------------------------------
        # Single-run mode: test against latest (or specified) run
        # --------------------------------------------------------------
        if run_id:
            with db.get_conn() as conn:
                target_run = conn.execute(
                    "SELECT * FROM runs WHERE run_id=?", (run_id,)
                ).fetchone()
            if not target_run:
                err_console.print(f"[red]Error:[/red] Run '{run_id}' not found.")
                sys.exit(1)
        else:
            target_run = db.get_latest_successful_run()
            if target_run is None:
                err_console.print(
                    f"[red]Error:[/red] No previous successful run found.\n"
                    f"Run [bold]pbt run[/bold] first, then [bold]pbt test[/bold]."
                )
                sys.exit(1)

        model_names = list(all_models.keys())
        model_outputs = db.get_model_outputs_from_run(target_run["run_id"], model_names)

        pretty_print.print_test_header(c, tests_dir, tests, target_run)

        test_results: list = []
        on_start, on_done = pretty_print.make_test_callbacks(c, test_results, total=len(tests))
        execute_tests(
            run_id=target_run["run_id"],
            tests=tests,
            model_outputs=model_outputs,
            storage_backend=db,
            on_test_start=on_start,
            on_test_done=on_done,
            llm_call=llm_call,
        )

        pretty_print.print_test_summary(c, test_results, target_run)

        failed  = sum(1 for r in test_results if r.status == "fail")
        errored = sum(1 for r in test_results if r.status == "error")
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
    except (FileNotFoundError, CyclicDependencyError, UnknownModelError) as exc:
        err_console.print(str(exc))
        sys.exit(1)

    console.print(pretty_print.models_table(list(models.values())))


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

    console.print(pretty_print.runs_table(rows))


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

    all_runs = db.get_latest_runs(limit=10_000)
    run_results: dict = {}
    for run in all_runs:
        run_results[run["run_id"]] = db.get_run_results(run["run_id"])

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
# pbt init  (command and scaffold templates defined in pbt/cli/init_files.py)
# ---------------------------------------------------------------------------

_register_init(main)


# ---------------------------------------------------------------------------
# pbt type-hints  (command defined in pbt/cli/type_hints.py)
# ---------------------------------------------------------------------------

_register_type_hints(main)


# ---------------------------------------------------------------------------
# pbt serve
# ---------------------------------------------------------------------------

@main.command("serve")
@click.option("--models-dir", default="models", show_default=True)
@click.option("--validation-dir", default="validation", show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--docs-output", default=".pbt/docs/index.html", show_default=True,
              help="Path to the pre-generated pbt docs HTML file.")
def serve(models_dir: str, validation_dir: str, host: str, port: int, docs_output: str) -> None:
    """Start the pbt HTTP server and open the docs page in the browser."""
    import threading
    import time
    import webbrowser
    from dotenv import load_dotenv
    load_dotenv()

    try:
        import uvicorn
    except ImportError:
        err_console.print("[red]Error:[/red] uvicorn is required. Install with: pip install uvicorn")
        sys.exit(1)

    try:
        from pbt.server.app import create_app
    except ImportError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    app = create_app(models_dir=models_dir, validation_dir=validation_dir)

    test_url = f"http://{host}:{port}/test"
    docs_path = Path(docs_output)
    if docs_path.exists():
        from fastapi.responses import HTMLResponse
        html_content = docs_path.read_text(encoding="utf-8")

        @app.get("/docs-report", response_class=HTMLResponse)
        def docs_report():  # noqa: ANN201
            return html_content

        docs_url = f"http://{host}:{port}/docs-report"
        console.print(f"[dim]Docs report:[/dim] {docs_url}")
        console.print(f"[dim]Test runner: [/dim] {test_url}")
    else:
        docs_url = test_url
        console.print(f"[dim]No docs file found at {docs_output}, opening test runner.[/dim]")

    console.print(f"[bold cyan]pbt serve[/bold cyan] → http://{host}:{port}")

    def _open_browser():
        time.sleep(0.8)
        webbrowser.open(docs_url)

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host=host, port=port)


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
