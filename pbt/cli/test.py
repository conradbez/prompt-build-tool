"""
pbt test — run test prompts from the tests/ directory against model outputs.

The `pbt test` Click command lives here and is attached to the main Click
group via :func:`register_command`, called from ``pbt/cli/__init__.py``.

Each test prompt has full Jinja2 context (``ref()`` works as in models). A test
passes when the LLM returns JSON containing ``{"results": "pass"}``.

Two modes:

* **Per-row** — when a ``promptparams.csv`` file is present (or inline
  ``--promptdata`` / ``--promptfile`` params are supplied), ``pbt run`` is
  executed once per row and tests are reported for each row individually.
* **Single-run** — otherwise tests run against the latest (or ``--run-id``)
  stored run's outputs.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
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
from pbt.promptparams import (
    load_promptparams,
    write_example,
    append_promptparams_row,
    parse_promptparams_row,
)
from pbt.cli import pretty_print
from pbt.cli.pretty_print import err_console


def register_command(main) -> None:
    """Attach the `pbt test` command to *main* Click group."""

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
    @click.option(
        "--promptdata",
        "promptdata",
        multiple=True,
        metavar="KEY=VALUE",
        help=(
            "Inline promptdata for this test run. When provided, pbt run is executed "
            "with these params and tests are reported against that run (an inline "
            "one-row promptparams). Repeatable: --promptdata tone=formal."
        ),
    )
    @click.option(
        "--promptfile",
        "promptfiles",
        multiple=True,
        metavar="NAME=PATH",
        help=(
            "Inline promptfile for this test run, paired with --promptdata. "
            "Repeatable: --promptfile doc=report.pdf."
        ),
    )
    @click.option(
        "--add-to-csv",
        is_flag=True,
        default=False,
        help=(
            "After running, append the inline --promptdata/--promptfile params as a "
            "new row in the promptparams file so the case is re-tested in future "
            "parameterised runs. Requires at least one --promptdata or --promptfile."
        ),
    )
    def test(
        models_dir: str,
        tests_dir: str,
        run_id: str | None,
        no_color: bool,
        promptparams_file: str,
        check_latest: bool,
        promptdata: tuple[str, ...],
        promptfiles: tuple[str, ...],
        add_to_csv: bool,
    ) -> None:
        """
        Run test prompts from the tests/ directory against model outputs.

        Each test prompt has full Jinja2 context (ref() works as in models).
        A test passes when the LLM returns JSON containing {"results": "pass"}.

        When a promptparams.csv file is present, pbt run is executed once per row
        and tests are reported for each row individually.
        Without promptparams, tests run against the latest (or specified) run.
        """
        from pbt.cli import _git_sha

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

        # Write the template next to the real promptparams file so users can
        # copy it into place (cp promptparams.csv.example promptparams.csv).
        example_path = Path(f"{promptparams_file}.example")
        try:
            write_example(example_path, dag_promptdata, dag_promptfiles)
            if dag_promptdata or dag_promptfiles:
                c.print(f"  [dim]promptparams.csv.example written → {example_path}[/dim]")
                c.print()
        except Exception:  # noqa: BLE001
            pass

        # ------------------------------------------------------------------
        # Parse inline --promptdata / --promptfile into a single synthetic row.
        # When present, this row drives a fresh run (an inline one-row
        # promptparams) and takes precedence over the CSV / --check-latest.
        # ------------------------------------------------------------------
        inline_row: dict[str, str] = {}
        for v in promptdata:
            if "=" not in v:
                err_console.print(f"[red]Error:[/red] --promptdata must be KEY=VALUE, got: {v!r}")
                sys.exit(1)
            k, _, val = v.partition("=")
            inline_row[f"promptdata.{k}"] = val
        for f in promptfiles:
            if "=" not in f:
                err_console.print(f"[red]Error:[/red] --promptfile must be NAME=PATH, got: {f!r}")
                sys.exit(1)
            k, _, val = f.partition("=")
            inline_row[f"promptfile.{k}"] = val

        if add_to_csv and not inline_row:
            err_console.print(
                "[red]Error:[/red] --add-to-csv requires at least one --promptdata or "
                "--promptfile to record as a new row."
            )
            sys.exit(1)

        # ------------------------------------------------------------------
        # Load promptparams rows (optional; skipped when --check-latest).
        # Inline params override everything else.
        # ------------------------------------------------------------------
        if inline_row:
            promptparams_rows = [inline_row]
        else:
            promptparams_rows = [] if check_latest else load_promptparams(promptparams_file)

        if promptparams_rows:
            # --------------------------------------------------------------
            # Per-row mode: run models then test for each CSV row
            # --------------------------------------------------------------
            if inline_row:
                c.print("  promptparams : [dim]inline --promptdata/--promptfile[/dim] (1 row)")
            else:
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
                    promptdata=row_promptdata or None,
                    promptfiles=row_promptfiles or None,
                    param_label=f"row_{idx}",
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

            # Snapshot the inline run's params into the promptparams file so the
            # case is re-tested in future parameterised runs.
            if add_to_csv and inline_row:
                row_promptdata, row_promptfiles = parse_promptparams_row(inline_row)
                append_promptparams_row(promptparams_file, row_promptdata, row_promptfiles)
                c.print(f"  [dim]added row → {promptparams_file}[/dim]")

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
