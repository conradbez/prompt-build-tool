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
from pbt.executor.graph import (
    load_models,
    execution_order,
    build_dag,
    compute_dag_hash,
    get_dag_promptdata,
    CyclicDependencyError,
    UnknownModelError,
    models_to_json,
    models_from_json,
)
from pbt.executor.executor import execute_run, ModelRunResult
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
        "Run only these models and their upstream dependencies. "
        "Unchanged nodes are served instantly from the prompt cache. "
        "Repeatable: -s tweet -s haiku"
    ),
)
@click.option(
    "--dag-id",
    default=None,
    metavar="HASH",
    help=(
        "Load the DAG snapshot from the database instead of reading "
        "*.prompt files from disk. Use the dag_hash shown after a previous run."
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
def run(models_dir: str, select: tuple[str, ...], dag_id: str | None, no_color: bool, promptdata: tuple[str, ...], promptfiles: tuple[str, ...], validation_dir: str) -> None:
    """Execute all prompt models in dependency order."""
    c = Console(highlight=not no_color)

    # Parse --promptdata KEY=VALUE pairs into a dict
    promptdata_vars: dict[str, str] = {}
    for v in promptdata:
        if "=" not in v:
            err_console.print(f"[red]Error:[/red] --promptdata must be KEY=VALUE, got: {v!r}")
            sys.exit(1)
        k, _, val = v.partition("=")
        promptdata_vars[k] = val

    # Parse --promptfile NAME=PATH pairs into a dict of open file objects.
    # Opening here (rather than passing paths) lets llm_call receive a ready-to-
    # read binary stream regardless of whether the caller is the CLI or the API.
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
    # Discover & validate models  (from disk or DB snapshot)
    # ------------------------------------------------------------------
    if dag_id:
        dag_json = db.load_dag(dag_id)
        if dag_json is None:
            err_console.print(
                f"[red]Error:[/red] DAG '{dag_id}' not found in database.\n"
                f"Run [bold]pbt run[/bold] without --dag-id first to register it."
            )
            sys.exit(1)
        all_models = models_from_json(dag_json)
        dag_hash = dag_id
    else:
        try:
            all_models = load_models(models_dir)
        except FileNotFoundError as exc:
            err_console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)
        dag_hash = compute_dag_hash(all_models)
        db.save_dag(dag_hash, models_to_json(all_models))

    try:
        ordered = execution_order(all_models)
    except (CyclicDependencyError, UnknownModelError) as exc:
        err_console.print(f"[red]Dependency error:[/red] {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # --select: run chosen models AND their full upstream dependency chain.
    # Unchanged nodes are served from the prompt cache automatically.
    # ------------------------------------------------------------------
    if select:
        for name in select:
            if name not in all_models:
                err_console.print(f"[red]Unknown model:[/red] '{name}'")
                sys.exit(1)

        selected_set = set(select)
        dag = build_dag(all_models)

        to_run: set[str] = set(selected_set)
        for name in selected_set:
            to_run.update(nx.ancestors(dag, name))

        ordered = [m for m in ordered if m.name in to_run]

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
    if select:
        c.print(f"  [dim](select: {sorted(select)})[/dim]")
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
        all_results = execute_run(
            run_id=run_id,
            ordered_models=ordered,
            on_model_start=on_start,
            on_model_done=on_done,
            llm_call=llm_call,
            rag_call=rag_call,
            promptdata=promptdata_vars or None,
            promptfiles=promptfiles_dict or None,
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

    # ------------------------------------------------------------------
    # Write outputs/ directory — one .md file per successful model
    # ------------------------------------------------------------------
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    written: list[str] = []
    for result in all_results:
        if result.status == "success" and result.llm_output:
            out_file = outputs_dir / f"{result.model_name}.md"
            out_file.write_text(result.llm_output, encoding="utf-8")
            written.append(result.model_name)

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
    # Resolve LLM backend
    # ------------------------------------------------------------------
    try:
        llm_call = resolve_llm_call(models_dir)
    except Exception as exc:
        err_console.print(f"[red]Backend resolution error:[/red] {exc}")
        sys.exit(1)

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
        llm_call=llm_call,
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
    table.add_column("promptdata() used", style="dim")
    table.add_column("File", style="dim")

    for i, model in enumerate(ordered, 1):
        deps = ", ".join(model.depends_on) if model.depends_on else "[dim]—[/dim]"
        promptdata_str = ", ".join(model.promptdata_used) if model.promptdata_used else "[dim]—[/dim]"
        table.add_row(str(i), model.name, deps, promptdata_str, str(model.path))

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
# pbt init
# ---------------------------------------------------------------------------

_INIT_FILES = {
    "basic-usage.md": """\
# Getting Started

| Directory | Purpose |
|-----------|---------|
| `models/` | <- START HERE: Prompt files — see `models/basic-usage.md` |
| `tests/` | LLM-as-judge tests — see `tests/basic-usage.md` |
| `validation/` | Pre-pass quality gates — see `validation/basic-usage.md` |
| `outputs/` | Generated outputs from `pbt run` (auto-created) |

Run: `pbt run` or `pbt run --promptdata topic="your topic"`
""",
    "tests/basic-usage.md": """\
# Tests

Used when you change prompts to make sure everything still works and passes quality standards.
Write prompts here that will run against your models to assess their quality.

Each `.prompt` file in this directory is an LLM-as-judge test.
The prompt should reference model outputs via `{{ ref('model_name') }}` and return JSON:
  - Pass: `{"results": "pass"}`
  - Fail: `{"results": "fail"}`

Example test file `tests/summary_has_bullets.prompt`:

    Does the following text contain at least 3 bullet points (lines starting with - or •)?

    {{ ref('summary') }}

    Reply with only valid JSON: {"results": "pass"} or {"results": "fail"}.
""",
    "models/basic-usage.md": """\
# Models

Write your prompts here. Each `.prompt` file defines one step in your pipeline.

You can:
- Reference other prompt outputs:  `{{ ref('other_prompt_name') }}`
- Access passed-in data:           `{{ promptdata("key") }}`
- Configure model behaviour:       `{{ config(output_format="json") }}`

Example chain — `models/topic.prompt` → `models/article.prompt` → `models/summary.prompt`:

    # topic.prompt
    Generate a catchy blog post topic about AI.

    # article.prompt
    Write a detailed article about: {{ ref('topic') }}

    # summary.prompt
    {{ config(output_format="json") }}
    Summarise this article in 3 bullet points. Return JSON: {"bullets": ["...", "...", "..."]}.
    {{ ref('article') }}

## client.py
Add a `models/client.py` to configure which LLM to call. It must expose a
`llm_call(prompt: str) -> str` function. See the scaffolded example for a
Gemini implementation.

Run with: `pbt run` or `pbt run --promptdata topic="your topic"`
""",
    "validation/basic-usage.md": """\
# Validation

Optional Python code that runs *before* an LLM prompt output is passed to the next prompt.
Use this to check basic quality gates and avoid wasting tokens on bad intermediate results.

Each `.py` file must expose a `validate(prompt: str, result: str) -> bool` function.
Returning `False` stops the pipeline and reports a validation error.

Example `validation/article.py`:

    def validate(prompt: str, result: str) -> bool:
        \"\"\"Article must be at least 200 characters and contain a markdown header.\"\"\"
        return len(result) >= 200 and "#" in result
""",
    "models/article.prompt": """\
{# Generate an article on a topic. Pass --promptdata topic=... or leave blank for a generic article. #}
{% if promptdata("topic") %}
Write a detailed, engaging article about: {{ promptdata("topic") }}
{% else %}
Write a detailed, engaging article about a fascinating topic of your choice.
{% endif %}

Structure it with an introduction, 3-4 body sections, and a conclusion.
Use markdown formatting with headers.
""",
    "models/summary.prompt": """\
Summarise the following article in 3 bullet points. Be concise.

{{ ref('article') }}
""",
    "tests/summary_has_bullets.prompt": """\
Does the following text contain at least 3 bullet points (lines starting with - or •)?

{{ ref('summary') }}

Reply with only valid JSON: {"results": "pass"} or {"results": "fail"}.
""",
}

_CLIENT_PY: dict[str, str] = {
    "gemini": """\
import os
from google import genai

def llm_call(prompt: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        contents=prompt,
    ).text
""",
    "openai": """\
import os
from openai import OpenAI

def llm_call(prompt: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
""",
    "anthropic": """\
import os
import anthropic

def llm_call(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=8096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
""",
}

_PROVIDERS = ("gemini", "openai", "anthropic")

@main.command("init")
@click.argument("project_name", default="generate_articles_example")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing files.")
@click.option(
    "--provider",
    type=click.Choice(_PROVIDERS, case_sensitive=False),
    default="gemini",
    show_default=True,
    help="LLM provider to use in the generated client.py.",
)
def init(project_name: str, force: bool, provider: str) -> None:
    """Scaffold a starter pbt project inside PROJECT_NAME/."""
    files = dict(_INIT_FILES)
    files["models/client.py"] = _CLIENT_PY[provider.lower()]
    files["validation/article.py"] = """\
def validate(prompt: str, result: str) -> bool:
    \"\"\"Article must be at least 200 characters and contain a markdown header.\"\"\"
    return len(result) >= 200 and "#" in result
"""

    root = Path(project_name)
    created, skipped = [], []

    for rel_path, content in files.items():
        path = root / rel_path
        if path.exists() and not force:
            skipped.append(str(path))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(str(path))

    for f in created:
        console.print(f"  [green]created[/green]  {f}")
    for f in skipped:
        console.print(f"  [dim]skipped[/dim]  {f}  [dim](use --force to overwrite)[/dim]")

    if created:
        console.print(f"\nRun [bold cyan]cd {project_name} && pbt run[/bold cyan], or [bold cyan]pbt run --promptdata topic='your topic'[/bold cyan]")


# ---------------------------------------------------------------------------
# pbt type-hints
# ---------------------------------------------------------------------------

@main.command("type-hints")
@click.option("--models-dir", default="models", show_default=True,
              help="Directory containing *.prompt files.")
@click.option("--validation-dir", default="validation", show_default=True,
              help="Directory containing per-model validation Python files.")
@click.option("--gen-dir", default=".pbt/gen", show_default=True,
              help="Output directory for generated context stubs.")
@click.option("--pyproject", "pyproject_path", default="pyproject.toml", show_default=True,
              help="Path to pyproject.toml to update with jinja-lsp config.")
def type_hints(models_dir: str, validation_dir: str, gen_dir: str, pyproject_path: str) -> None:
    """Generate jinja-lsp context stubs for autocomplete in .prompt templates.

    Creates one <model>_context.py stub per model in GEN_DIR, importing
    the validation class of each upstream dependency so that jinja-lsp can
    offer typed autocomplete inside the template.

    Also ensures pyproject.toml contains a [tool.jinja-lsp] section pointing
    at your models/, validation/, and the generated stubs directory.
    """
    from pbt.type_hints import generate_stubs, update_pyproject_toml

    try:
        written = generate_stubs(models_dir, validation_dir, gen_dir)
    except FileNotFoundError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    for path in written:
        console.print(f"  [green]wrote[/green]  {path}")

    toml_updated = update_pyproject_toml(models_dir, validation_dir, gen_dir, pyproject_path)
    if toml_updated:
        console.print(f"  [green]wrote[/green]  {pyproject_path}  [dim](added [tool.jinja-lsp] section)[/dim]")
    else:
        console.print(f"  [dim]skipped[/dim] {pyproject_path}  [dim]([tool.jinja-lsp] already present)[/dim]")

    console.print(
        f"\n[bold]Done.[/bold] Install the [cyan]jinja-lsp[/cyan] VS Code extension "
        f"to activate autocomplete in your .prompt files."
    )


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
        from utils.server.app import create_app
    except ImportError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    app = create_app(models_dir=models_dir, validation_dir=validation_dir)

    # Mount the pre-generated docs HTML as a static route if the file exists
    docs_path = Path(docs_output)
    if docs_path.exists():
        from fastapi.responses import HTMLResponse
        html_content = docs_path.read_text(encoding="utf-8")
        api_url = f"http://{host}:{port}/docs"
        api_link = (
            f'<nav style="'
            f"font-family:sans-serif;font-size:13px;background:#1e1e2e;color:#cdd6f4;"
            f"padding:0 20px;display:flex;align-items:center;gap:24px;height:40px;"
            f'box-shadow:0 1px 4px rgba(0,0,0,.4);position:sticky;top:0;z-index:999">'
            f'<span style="font-weight:600;letter-spacing:.5px">pbt</span>'
            f'<a href="{api_url}" style="color:#89b4fa;text-decoration:none" '
            f'target="_blank">API docs ↗</a>'
            f"</nav>"
        )
        html_content = html_content.replace("<body>", f"<body>\n{api_link}", 1)

        @app.get("/docs-report", response_class=HTMLResponse)
        def docs_report():  # noqa: ANN201
            return html_content

        docs_url = f"http://{host}:{port}/docs-report"
        console.print(f"[dim]Docs report:[/dim] {docs_url}")
        console.print(f"[dim]API docs:    [/dim] {api_url}")
    else:
        docs_url = f"http://{host}:{port}/docs"
        console.print(f"[dim]No docs file found at {docs_output}, opening API docs.[/dim]")

    console.print(f"[bold cyan]pbt serve[/bold cyan] → http://{host}:{port}")

    def _open_browser():
        time.sleep(0.8)  # give uvicorn a moment to start
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


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "—"
    return str(ts)[:19].replace("T", " ")
