"""
promptmap CLI

Commands
--------
promptmap init         Scaffold a starter project (or add secrets/data/files YAML).
promptmap run          Execute all prompt models (or a subset via --select).
promptmap test         Run test prompts from the tests/ directory.
promptmap ls           List discovered models and their dependencies.
promptmap show-runs    Show recent run history from the SQLite store.
promptmap show-result  Print the stored output for a specific model + run.
promptmap docs         Generate an HTML report of past runs.
promptmap serve        Serve the docs report and a web/HTTP model runner.
promptmap type-hints   Write jinja-lsp autocomplete stubs for .prompt files.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import click
import networkx as nx
from rich.console import Console
from rich.markup import escape

from promptmap import db
from promptmap.storage.sqlite import StaleDatabaseError
from promptmap.executor.graph import (
    load_models,
    build_dag,
    get_dag_promptdata,
    CyclicDependencyError,
    UnknownModelError,
    UnknownModelTypeError,
)
from promptmap.executor.executor import execute_run
from promptmap.gates import execute_gated_run, gate_reads, gate_upstream, load_gates
from promptmap.llm import (
    resolve_blob_store,
    resolve_classify_call,
    resolve_llm_call,
    resolve_test_judge,
    try_load_client_module,
)
from promptmap.rag import resolve_rag_call
from promptmap.run_ids import record_run_ids
from promptmap.secrets import MissingSecretsError
from promptmap.project_values import (
    PROMPTDATA_FILE,
    PROMPTFILES_FILE,
    SECRETS_FILE,
    ProjectValuesError,
    load_values,
)
from promptmap.global_instruction import resolve_global_instruction
from promptmap.docs import generate_docs
from promptmap.files import FileOutputError, contains_files, decode_output, display_text, export_files, iter_files
from promptmap.validator import load_validators
from promptmap.cli.vscode import is_running_in_vscode, setup_vscode_associations
from promptmap.cli.type_hints import register_command as _register_type_hints, generate_stubs as _generate_stubs
from promptmap.cli.init_files import register_command as _register_init
from promptmap.cli.test import register_command as _register_test
from promptmap.cli import pretty_print
from promptmap.cli.pretty_print import console, err_console


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

def use_client_blob_store(map_dir: str) -> None:
    """Point the run database at the blob store client.py declares, if any."""
    try:
        store = resolve_blob_store(map_dir)
    except TypeError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    if store is not None:
        db.set_blob_store(store)


def init_db_or_exit() -> None:
    """Open the run database, reporting a pre-schema one as advice, not a crash."""
    try:
        db.init_db()
    except StaleDatabaseError as exc:
        err_console.print("[red]Outdated promptmap database:[/red]")
        # soft_wrap keeps the suggested shell commands on one line, copy-pastable.
        err_console.print(str(exc), soft_wrap=True, highlight=False)
        sys.exit(1)


@click.group(epilog="Run `promptmap COMMAND --help` for a command's options.")
@click.version_option()
def main() -> None:
    """promptmap — dbt-inspired LLM prompt orchestration.

    Models are Jinja2 *.prompt files in map/ that ref() each other; promptmap
    runs them in dependency order, caches LLM calls and stores every run.

    \b
    Get started:
      promptmap init my_project   scaffold a project, then cd my_project
      promptmap run               run every model
      promptmap test              check outputs with test prompts
      promptmap docs --open       browse past runs
    """


# ---------------------------------------------------------------------------
# promptmap run
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--map-dir",
    default="map",
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
        "Overrides promptdata.yml. "
        "Repeatable: --promptdata country=USA --promptdata tone=formal"
    ),
)
@click.option(
    "--secret",
    "secret_flags",
    multiple=True,
    metavar="NAME=VALUE",
    help=(
        "Pass a credential to model kinds (call.secrets) and llm_call. A bare "
        "NAME reads the environment variable NAME. Never rendered, cached or "
        "stored. Overrides secrets.yml. "
        "Repeatable: --secret OPENAI_API_KEY=sk-... --secret MODAL_TOKEN_ID"
    ),
)
@click.option(
    "--promptfile",
    "promptfiles",
    multiple=True,
    metavar="NAME=PATH",
    help=(
        "Provide a file by name for models that declare it in their config block. "
        "Overrides promptfiles.yml. Repeatable: --promptfile doc=report.pdf --promptfile img=chart.png"
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
@click.option(
    "--global-instruction",
    "global_instruction_path",
    default=None,
    metavar="PATH",
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Prompt file rendered into every model's prompt. "
        "Overrides the global.prompt discovered alongside --map-dir."
    ),
)
@click.option(
    "--tests-dir",
    default="tests",
    show_default=True,
    help=(
        "Directory holding gate tests: every test there runs right after the "
        "models it reads, and a failing one stops the run. Opt a test out "
        "with {{ config(gate=False) }}."
    ),
)
def run(map_dir: str, select: tuple[str, ...], no_color: bool, promptdata: tuple[str, ...], secret_flags: tuple[str, ...], promptfiles: tuple[str, ...], validation_dir: str, clear_cache: bool, global_instruction_path: str | None, tests_dir: str) -> None:
    """Execute all prompt models in dependency order.

    Tests in the tests dir are gates unless they set {{ config(gate=False) }}:
    they run first, after only the models they read, and the rest of the
    pipeline runs if they all pass.

    Defaults for any value not passed as a flag are read from promptdata.yml,
    promptfiles.yml and secrets.yml in the current directory (create them
    with `promptmap init --data --files --secrets`). Flags override them key by key.

    \b
    Examples:
      promptmap run
      promptmap run -s summaries --promptdata topic=cats
      promptmap run --promptfile doc=report.pdf --secret OPENAI_API_KEY
    """
    c = Console(highlight=not no_color)

    if clear_cache:
        init_db_or_exit()
        cleared = db.clear_cache()
        c.print(f"  [yellow]Cache cleared[/yellow] ({cleared} entr{'ies' if cleared != 1 else 'y'} invalidated)\n")

    # promptdata.yml / promptfiles.yml in the project root supply defaults;
    # --promptdata / --promptfile override them key by key.
    try:
        promptdata_vars: dict[str, str] = load_values(PROMPTDATA_FILE)
        file_promptfiles: dict[str, str] = load_values(PROMPTFILES_FILE)
    except ProjectValuesError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        sys.exit(1)
    for label, values in ((PROMPTDATA_FILE, promptdata_vars), (PROMPTFILES_FILE, file_promptfiles)):
        if values:
            c.print(f"  [dim]{label}: {', '.join(values)}[/dim]")

    # Parse --promptdata KEY=VALUE pairs into a dict
    for v in promptdata:
        if "=" not in v:
            err_console.print(f"[red]Error:[/red] --promptdata must be KEY=VALUE, got: {v!r}")
            sys.exit(1)
        k, _, val = v.partition("=")
        promptdata_vars[k] = val

    secrets = _secrets_or_exit(secret_flags)

    # Parse --promptfile NAME=PATH pairs into a dict of open file objects.
    promptfiles_dict: dict = {}
    promptfile_paths: dict[str, str] = dict(file_promptfiles)  # gate tests open their own copies
    for f in promptfiles:
        if "=" not in f:
            err_console.print(f"[red]Error:[/red] --promptfile must be NAME=PATH, got: {f!r}")
            sys.exit(1)
        k, _, val = f.partition("=")
        promptfile_paths[k] = val
    for k, val in promptfile_paths.items():
        try:
            promptfiles_dict[k] = open(val, "rb")  # noqa: WPS515  – closed by llm_call consumer
        except OSError as exc:
            err_console.print(f"[red]Error:[/red] Cannot open promptfile '{val}': {exc}")
            sys.exit(1)

    init_db_or_exit()

    # ------------------------------------------------------------------
    # Discover & validate models
    # ------------------------------------------------------------------
    try:
        try_load_client_module(map_dir)  # registers any project-local model types
        all_models = load_models(map_dir)
    except (FileNotFoundError, UnknownModelTypeError) as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        sys.exit(1)
    use_client_blob_store(map_dir)
    try:
        dag = build_dag(all_models)
    except (CyclicDependencyError, UnknownModelError) as exc:
        err_console.print(f"[red]Dependency error:[/red] {exc}")
        sys.exit(1)
    # ------------------------------------------------------------------
    # --select: run chosen models AND their full upstream dependency chain.
    # ------------------------------------------------------------------
    if select:
        for name in select:
            if name not in all_models:
                err_console.print(f"[red]Unknown model:[/red] '{name}'")
                sys.exit(1)
        to_run: set[str] = set(select)
        for name in select:
            to_run.update(nx.ancestors(dag, name))
        ordered = [all_models[name] for name in to_run]
    else:
        ordered = list(all_models.values())

    # ------------------------------------------------------------------
    # Gate tests: the ones whose models are all part of this run.
    # ------------------------------------------------------------------
    run_names = {m.name for m in ordered}
    try:
        gates = {
            name: source
            for name, source in load_gates(tests_dir).items()
            if set(gate_reads(name, source, all_models)) <= run_names
        }
    except UnknownModelError as exc:
        err_console.print(f"[red]Dependency error:[/red] {exc}")
        sys.exit(1)

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

    # Discover user-provided client.py and rag.py from map_dir
    try:
        llm_call = resolve_llm_call(map_dir)
        rag_call = resolve_rag_call(map_dir)
        if global_instruction_path:
            global_instruction = Path(global_instruction_path).read_text(encoding="utf-8")
        else:
            global_instruction = resolve_global_instruction(map_dir)
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

    if global_instruction:
        source_label = global_instruction_path or "global.prompt"
        c.print(f"  Global instruction: [cyan]{source_label}[/cyan]")
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

    record_run_ids(
        db, run_id, all_models.values(), ordered,
        tests_dir=tests_dir, global_instruction=global_instruction,
        promptdata=promptdata_vars, promptfiles=promptfiles_dict,
    )

    run_kwargs = dict(
        storage_backend=db,
        on_model_start=on_start,
        on_model_done=on_done,
        llm_call=llm_call,
        rag_call=rag_call,
        promptdata=promptdata_vars or None,
        promptfiles=promptfiles_dict or None,
        validators=validators or None,
        global_instruction=global_instruction,
        secrets=secrets,
    )
    gated = None
    try:
        if gates:
            try:
                judge = resolve_test_judge(map_dir)
                classify_call = resolve_classify_call(map_dir)
            except Exception as exc:
                err_console.print(f"[red]Backend resolution error:[/red] {exc}")
                db.finish_run(run_id, "error")
                sys.exit(1)
            gate_results: list = []
            on_test_start, on_test_done = pretty_print.make_test_callbacks(c, gate_results, total=len(gates))
            gate_models = gate_upstream(gates, all_models, dag)
            remaining = sum(1 for m in ordered if m.name not in gate_models)
            gated = asyncio.run(execute_gated_run(
                run_id=run_id,
                ordered_models=ordered,
                gates=gates,
                gate_models=gate_models,
                on_gates_start=lambda first: pretty_print.print_gates_header(c, gates, first),
                on_test_start=on_test_start,
                on_test_done=on_test_done,
                on_gates_done=lambda result: pretty_print.print_gates_done(c, result, remaining),
                judge=judge,
                classify_call=classify_call,
                test_promptfiles=promptfile_paths or None,
                **run_kwargs,
            ))
            all_results = gated.model_results
        else:
            all_results = asyncio.run(execute_run(run_id=run_id, ordered_models=ordered, **run_kwargs))
    except (EnvironmentError, MissingSecretsError) as exc:
        err_console.print(f"\n[red]Configuration error:[/red] {exc}")
        db.finish_run(run_id, "error")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    errors = sum(1 for r in all_results if r.status == "error")
    successes = sum(1 for r in all_results if r.status == "success")

    final_status = "success" if errors == 0 else ("partial" if successes > 0 else "error")
    gate_failed = gated is not None and bool(gated.blocked or not gated.gates_passed)
    if gate_failed:
        final_status = "error"
    db.finish_run(run_id, final_status)

    # Write outputs/ directory — one file per successful model.
    # Extension comes from {{ config(output_extension="html") }}; defaults to "md".
    # A model that produced files gets them in outputs/<model>/, and its text
    # part (if any) in the usual outputs/<model>.<ext>.
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    written: list[str] = []
    for result in all_results:
        if result.status == "success" and result.llm_output:
            value = result.value
            if contains_files(value):
                files_dir = outputs_dir / result.model_name
                if files_dir.is_dir():
                    shutil.rmtree(files_dir)  # last run's files for this model
                export_files(value, files_dir)
                text = display_text(value)
            else:
                text = value if isinstance(value, str) else result.llm_output
            if text:
                ext = all_models[result.model_name].config.get("output_extension", "md")
                out_file = outputs_dir / f"{result.model_name}.{ext}"
                out_file.write_text(text, encoding="utf-8")
            written.append(result.model_name)

    pretty_print.print_run_summary(c, all_results, outputs_dir, written, run_id)

    if gate_failed:
        pretty_print.print_gate_failure(c, gated)
        sys.exit(1)
    if errors:
        sys.exit(1)

    try:
        _generate_stubs(validation_dir)
    except Exception:
        pass  # type-hint generation is best-effort; never block a run


# ---------------------------------------------------------------------------
# promptmap test  (command defined in promptmap/cli/test.py)
# ---------------------------------------------------------------------------

_register_test(main)


# ---------------------------------------------------------------------------
# promptmap ls
# ---------------------------------------------------------------------------

@main.command("ls", short_help="List models, their dependencies and inputs.")
@click.option("--map-dir", default="map", show_default=True, help="Directory containing *.prompt files.")
def list_models(map_dir: str) -> None:
    """List all discovered models, their dependencies and promptdata() keys."""
    try:
        try_load_client_module(map_dir)  # registers any project-local model types
        models = load_models(map_dir)
    except (FileNotFoundError, CyclicDependencyError, UnknownModelError, UnknownModelTypeError) as exc:
        err_console.print(escape(str(exc)))
        sys.exit(1)

    console.print(pretty_print.models_table(list(models.values())))


# ---------------------------------------------------------------------------
# promptmap show-runs
# ---------------------------------------------------------------------------

@main.command("show-runs")
@click.option("--limit", default=10, show_default=True, help="Number of runs to show.")
def show_runs(limit: int) -> None:
    """Show recent run history."""
    init_db_or_exit()
    rows = db.get_latest_runs(limit)

    if not rows:
        console.print("[dim]No runs recorded yet. Run `promptmap run` first.[/dim]")
        return

    console.print(pretty_print.runs_table(rows))


# ---------------------------------------------------------------------------
# promptmap show-result
# ---------------------------------------------------------------------------

@main.command("show-result", short_help="Print the stored output for one model.")
@click.argument("model_name")
@click.option("--run-id", default=None, help="Specific run ID (defaults to latest).")
@click.option(
    "--show",
    type=click.Choice(["output", "prompt", "all"]),
    default="output",
    show_default=True,
    help="output: the (validated) LLM output; prompt: the rendered prompt sent; all: both.",
)
@click.option("--map-dir", default="map", show_default=True,
              help="Used to find client.py, for a custom blob store.")
@click.option("--save-files", "save_files", default=None, metavar="DIR",
              help="Write the files this model produced into DIR.")
def show_result(model_name: str, run_id: str | None, show: str, map_dir: str, save_files: str | None) -> None:
    """Print stored output for MODEL_NAME (from the latest run by default).

    \b
    Examples:
      promptmap show-result summaries
      promptmap show-result summaries --show all --run-id <id from promptmap show-runs>
    """
    init_db_or_exit()
    use_client_blob_store(map_dir)

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

    stored = row["llm_output_validated"] or row["llm_output"]
    try:
        value = decode_output(stored, db.blob_store())
    except FileOutputError as exc:
        err_console.print(f"[red]Cannot read stored files:[/red] {exc}")
        value = stored

    if show in ("output", "all"):
        console.rule("[dim]LLM output[/dim]")
        if contains_files(value):
            text = display_text(value)
            if text:
                console.print(text, markup=False, highlight=False)
            console.rule("[dim]Files[/dim]")
            for label, file in iter_files(value):
                console.print(f"{label}  [dim]{file.mime}, {file.size} bytes, sha256:{file.sha256[:12]}[/dim]")
        else:
            console.print(value or "", markup=False, highlight=False)

    if save_files:
        if not contains_files(value):
            err_console.print(f"Model '{model_name}' produced no files in this run.")
            sys.exit(1)
        for path in export_files(value, save_files):
            console.print(f"[green]saved[/green] {path}")

    if row["error"]:
        console.rule("[red]Error[/red]")
        console.print(row["error"])


# ---------------------------------------------------------------------------
# promptmap docs
# ---------------------------------------------------------------------------

@main.command("docs")
@click.option(
    "--map-dir",
    default="map",
    show_default=True,
    help="Directory containing *.prompt files (for DAG diagram).",
)
@click.option(
    "--output",
    default=".promptmap/docs/index.html",
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
def docs(map_dir: str, output: str, open_browser: bool) -> None:
    """Generate a self-contained HTML report of all previous runs."""
    import webbrowser

    init_db_or_exit()

    all_runs = db.get_latest_runs(limit=10_000)
    run_results: dict = {}
    for run in all_runs:
        run_results[run["run_id"]] = db.get_run_results(run["run_id"])

    models = None
    try:
        try_load_client_module(map_dir)  # registers any project-local model types
        models = load_models(map_dir)
    except (FileNotFoundError, Exception):
        pass
    use_client_blob_store(map_dir)

    output_path = Path(output)
    generate_docs(
        runs=list(all_runs),
        run_results=run_results,
        models=models,
        output_path=output_path,
        blob_store=db.blob_store(),
    )

    console.print(f"[green]Docs generated:[/green] [bold]{output_path}[/bold]")

    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())


# ---------------------------------------------------------------------------
# promptmap init  (command and scaffold templates defined in promptmap/cli/init_files.py)
# ---------------------------------------------------------------------------

_register_init(main)


# ---------------------------------------------------------------------------
# promptmap type-hints  (command defined in promptmap/cli/type_hints.py)
# ---------------------------------------------------------------------------

_register_type_hints(main)


# ---------------------------------------------------------------------------
# promptmap serve
# ---------------------------------------------------------------------------

@main.command("serve", short_help="Serve the docs report and a web/HTTP model runner.")
@click.option("--map-dir", default="map", show_default=True, help="Directory containing *.prompt files.")
@click.option("--validation-dir", default="validation", show_default=True,
              help="Directory containing per-model validation Python files.")
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind.")
@click.option("--port", default=8000, show_default=True, help="Port to listen on.")
@click.option("--docs-output", default=".promptmap/docs/index.html", show_default=True,
              help="Path to the pre-generated promptmap docs HTML file.")
@click.option("--secret", "secret_flags", multiple=True, metavar="NAME=VALUE",
              help="Credential every run receives; a bare NAME reads that env var. "
                   "Overrides secrets.yml. Repeatable.")
def serve(map_dir: str, validation_dir: str, host: str, port: int, docs_output: str, secret_flags: tuple[str, ...]) -> None:
    """Start the promptmap HTTP server and open the docs page in the browser.

    Serves the report from `promptmap docs` (if generated) and a /run page for
    running models over HTTP. Secrets come from secrets.yml and --secret.
    """
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
        from promptmap.server.app import create_app
    except ImportError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    secrets = _secrets_or_exit(secret_flags)
    app = create_app(map_dir=map_dir, validation_dir=validation_dir, secrets=secrets)

    test_url = f"http://{host}:{port}/run"
    docs_path = Path(docs_output)
    if docs_path.exists():
        from fastapi.responses import HTMLResponse
        html_content = docs_path.read_text(encoding="utf-8")

        @app.get("/docs-report", response_class=HTMLResponse)
        def docs_report():  # noqa: ANN201
            return html_content

        # Files models produced, exported by `promptmap docs` beside the report and
        # linked as files/<sha256>/<name> — relative to /docs-report, so /files.
        _serve_docs_files(app, docs_path.parent / "files")

        docs_url = f"http://{host}:{port}/docs-report"
        console.print(f"[dim]Docs report:[/dim] {docs_url}")
        console.print(f"[dim]Test runner: [/dim] {test_url}")
    else:
        docs_url = test_url
        console.print(f"[dim]No docs file found at {docs_output}, opening test runner.[/dim]")

    console.print(f"[bold cyan]promptmap serve[/bold cyan] → http://{host}:{port}")

    def _open_browser():
        time.sleep(0.8)
        webbrowser.open(docs_url)

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _secrets_or_exit(flags: tuple[str, ...]):
    """Parse --secret flags over secrets.yml, or exit naming the one that has no value."""
    from dotenv import load_dotenv

    from promptmap.secrets import Secrets, parse_secret_flags

    load_dotenv()  # a bare --secret NAME may be defined in .env
    try:
        # secrets.yml in the project root supplies defaults; --secret wins.
        return Secrets({**load_values(SECRETS_FILE), **parse_secret_flags(flags)})
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


def _serve_docs_files(app, files_dir: Path) -> None:
    """Serve the report's exported files without letting them act as pages.

    Model output is untrusted, and it shares this server's origin.  Images the
    report previews are served as images; everything else is a download.  A
    sandboxing CSP covers both, so even an SVG opened directly cannot run
    script.
    """
    import mimetypes
    import re

    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    from promptmap.docs import PREVIEW_MIMES
    from promptmap.files import safe_name

    @app.get("/files/{sha256}/{name}", include_in_schema=False)
    def docs_file(sha256: str, name: str):  # noqa: ANN202
        try:
            safe_name(name)
        except FileOutputError:
            raise HTTPException(status_code=404) from None
        path = files_dir / sha256 / name
        if not re.fullmatch(r"[0-9a-f]{64}", sha256) or not path.is_file():
            raise HTTPException(status_code=404)
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        inline = mime in PREVIEW_MIMES
        return FileResponse(
            path,
            media_type=mime if inline else "application/octet-stream",
            headers={
                "Content-Disposition": "inline" if inline else "attachment",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
                "X-Content-Type-Options": "nosniff",
            },
        )


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None
