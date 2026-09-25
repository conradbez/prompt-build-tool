"""
pbt test — run test prompts from the tests/ directory against model outputs.

The `pbt test` Click command lives here and is attached to the main Click
group via :func:`register_command`, called from ``pbt/cli/__init__.py``.

Each test prompt has full Jinja2 context (``ref()`` works as in models). A test
passes when the LLM returns JSON containing ``{"results": "pass"}``.

Two modes:

* **Per-case** — when promptparams YAML cases are found (``promptparams.yml``,
  ``promptparams/*.yml``, or ``--promptparams``), or inline ``--promptdata`` /
  ``--promptfile`` params are supplied, ``pbt run`` is executed once per case
  and tests are reported for each named case individually.
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
from pbt.global_instruction import resolve_global_instruction
from pbt.llm import resolve_llm_call, try_load_client_module
from pbt.rag import resolve_rag_call
from pbt.tester import load_tests, execute_tests
from pbt.promptparams import (
    EXAMPLE_PATH,
    PromptParamsError,
    build_case,
    load_baselines,
    load_cases,
    save_case,
    write_example,
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
        help="Use outputs from this specific run (default: latest run). Ignored when promptparams cases are found.",
    )
    @click.option("--no-color", is_flag=True, default=False)
    @click.option(
        "--promptparams",
        "promptparams_paths",
        multiple=True,
        metavar="PATH",
        help=(
            "YAML file or directory of test cases (repeatable; files are combined). "
            "Default: promptparams.yml and promptparams/*.yml when they exist. "
            "When cases are found, pbt run is executed for each case and tests "
            "are reported per case."
        ),
    )
    @click.option(
        "--case",
        "case_filters",
        multiple=True,
        metavar="NAME",
        help="Only run cases whose name matches (exact or glob, case-insensitive). Repeatable.",
    )
    @click.option(
        "--check-latest",
        is_flag=True,
        default=False,
        help="Skip promptparams cases and test against the latest stored run instead.",
    )
    @click.option(
        "--promptdata",
        "promptdata",
        multiple=True,
        metavar="KEY=VALUE",
        help=(
            "Inline promptdata for this test run. When provided, pbt run is executed "
            "with these params and tests are reported against that run (an inline "
            "one-case promptparams, which inherits the 'default' baseline like any "
            "case). Repeatable: --promptdata tone=formal."
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
        "--extends",
        "extends",
        multiple=True,
        metavar="BASELINE",
        help=(
            "Baseline(s) the inline --promptdata/--promptfile case builds on, "
            "instead of 'default'. Repeatable."
        ),
    )
    @click.option(
        "--save-case",
        "save_case_name",
        default=None,
        metavar="NAME",
        help=(
            "After running, save the inline --promptdata/--promptfile params as a "
            "named case in promptparams/<name>.yml so it is re-tested in future "
            "parameterised runs. Requires at least one --promptdata or --promptfile."
        ),
    )
    def test(
        models_dir: str,
        tests_dir: str,
        run_id: str | None,
        no_color: bool,
        promptparams_paths: tuple[str, ...],
        case_filters: tuple[str, ...],
        check_latest: bool,
        promptdata: tuple[str, ...],
        promptfiles: tuple[str, ...],
        extends: tuple[str, ...],
        save_case_name: str | None,
    ) -> None:
        """
        Run test prompts from the tests/ directory against model outputs.

        Each test prompt has full Jinja2 context (ref() works as in models).
        A test passes when the LLM returns JSON containing {"results": "pass"}.

        When promptparams YAML cases are found, pbt run is executed once per
        case and tests are reported for each named case individually.
        Without promptparams, tests run against the latest (or specified) run.
        """
        from pbt.cli import _git_sha, init_db_or_exit

        c = Console(highlight=not no_color)
        init_db_or_exit()

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
            try_load_client_module(models_dir)  # registers any project-local model types
            all_models = load_models(models_dir)
        except FileNotFoundError as exc:
            err_console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)

        from pbt.cli import use_client_blob_store
        use_client_blob_store(models_dir)

        # ------------------------------------------------------------------
        # Resolve LLM backend (needed for both modes)
        # ------------------------------------------------------------------
        try:
            llm_call = resolve_llm_call(models_dir)
            rag_call = resolve_rag_call(models_dir)
            # Models under test render exactly as they do in a real run.  The
            # test prompts themselves never get it — a judge given style
            # instructions is a biased judge.
            global_instruction = resolve_global_instruction(models_dir)
        except Exception as exc:
            err_console.print(f"[red]Backend resolution error:[/red] {exc}")
            sys.exit(1)

        # ------------------------------------------------------------------
        # Write promptparams.yml.example — a template for this DAG's inputs
        # ------------------------------------------------------------------
        from pbt.executor.parser_initial import detect_used_promptdata

        dag_promptdata = get_dag_promptdata(all_models)
        for src in tests.values():
            for key in detect_used_promptdata(src):
                if key not in dag_promptdata:
                    dag_promptdata.append(key)
        dag_promptfiles = get_dag_promptfiles(all_models)

        # Not a *.yml file, so it is never loaded as cases itself; users copy
        # it into place (cp promptparams.yml.example promptparams.yml).
        example_path = Path(EXAMPLE_PATH)
        try:
            write_example(example_path, dag_promptdata, dag_promptfiles)
            if dag_promptdata or dag_promptfiles:
                c.print(f"  [dim]promptparams example written → {example_path}[/dim]")
                c.print()
        except Exception:  # noqa: BLE001
            pass

        # ------------------------------------------------------------------
        # Parse inline --promptdata / --promptfile into a single case.  When
        # present, it drives a fresh run and takes precedence over the YAML
        # cases / --check-latest.  It still inherits baselines like any case.
        # ------------------------------------------------------------------
        inline_data: dict[str, str] = {}
        inline_files: dict[str, str] = {}
        for v in promptdata:
            if "=" not in v:
                err_console.print(f"[red]Error:[/red] --promptdata must be KEY=VALUE, got: {v!r}")
                sys.exit(1)
            k, _, val = v.partition("=")
            inline_data[k] = val
        for f in promptfiles:
            if "=" not in f:
                err_console.print(f"[red]Error:[/red] --promptfile must be NAME=PATH, got: {f!r}")
                sys.exit(1)
            k, _, val = f.partition("=")
            inline_files[k] = val
        inline = bool(inline_data or inline_files)

        if save_case_name is not None and not inline:
            err_console.print(
                "[red]Error:[/red] --save-case requires at least one --promptdata or "
                "--promptfile to record as a new case."
            )
            sys.exit(1)
        if extends and not inline:
            err_console.print(
                "[red]Error:[/red] --extends applies to inline --promptdata/--promptfile params."
            )
            sys.exit(1)

        _reject_csv(promptparams_paths)

        # ------------------------------------------------------------------
        # Load promptparams cases (optional; skipped when --check-latest).
        # ------------------------------------------------------------------
        inline_spec: dict = {"promptdata": inline_data, "promptfiles": inline_files}
        if extends:
            inline_spec["extends"] = list(extends)
        try:
            if inline:
                cases = [build_case(
                    save_case_name or "inline",
                    inline_spec,
                    load_baselines(promptparams_paths),
                    where="inline params",
                )]
            elif check_latest:
                cases = []
            else:
                cases = _filter_cases(load_cases(promptparams_paths), case_filters)
        except PromptParamsError as exc:
            err_console.print(f"[red]promptparams error:[/red] {exc}")
            sys.exit(1)

        if case_filters and not cases and not inline and not check_latest:
            err_console.print(
                f"[red]Error:[/red] no promptparams case matches {', '.join(case_filters)}."
            )
            sys.exit(1)

        if cases:
            # --------------------------------------------------------------
            # Per-case mode: run models then test for each case
            # --------------------------------------------------------------
            if inline:
                c.print("  promptparams : [dim]inline --promptdata/--promptfile[/dim] (1 case)")
            else:
                c.print(
                    f"  promptparams : {len(cases)} case{'s' if len(cases) != 1 else ''}"
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

            for idx, case in enumerate(cases, start=1):
                row_promptdata, row_promptfiles = case.promptdata, case.promptfiles
                c.rule(f"[bold]Case {idx}/{len(cases)}[/bold] — {case.name}")
                inputs = [f"{k}={v}" for k, v in row_promptdata.items()]
                inputs += [f"{k}={v}" for k, v in row_promptfiles.items()]
                if inputs:
                    c.print(f"  [dim]{_truncate(', '.join(inputs))}[/dim]")

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
                    global_instruction=global_instruction,
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
                    param_label=case.name,
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
            n_cases = f"{len(cases)} case{'s' if len(cases) != 1 else ''}"
            c.print(f"  [{overall_color}]{total_passed}/{len(all_test_results)} passed across {n_cases}[/{overall_color}]")

            # Save the inline params as a named case so it is re-tested in
            # future parameterised runs.  Only what was passed inline is
            # written, so the saved case keeps inheriting its baselines.
            if save_case_name is not None:
                try:
                    saved = save_case(
                        "promptparams",
                        save_case_name,
                        inline_data,
                        inline_files,
                        extends=list(extends) or None,
                    )
                except PromptParamsError as exc:
                    err_console.print(f"[red]Error:[/red] {exc}")
                    sys.exit(1)
                c.print(f"  [dim]saved case '{save_case_name}' → {saved}[/dim]")

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


def _reject_csv(paths: tuple[str, ...]) -> None:
    """Stop with a pointer to the YAML format when only a legacy CSV is around."""
    from pbt.promptparams import find_promptparams_files

    legacy = [p for p in paths if p.endswith(".csv")]
    if not paths and Path("promptparams.csv").exists():
        try:
            if not find_promptparams_files():
                legacy = ["promptparams.csv"]
        except PromptParamsError:
            pass
    if legacy:
        err_console.print(
            f"[red]Error:[/red] {legacy[0]}: promptparams are now YAML. Move each row to a "
            "named case in promptparams.yml, e.g.\n\n"
            "  cases:\n"
            "    - name: Formal tone\n"
            "      promptdata:\n"
            "        tone: formal\n"
            "      promptfiles:\n"
            "        document: report.pdf\n\n"
            "See the README section on promptparams."
        )
        sys.exit(1)


def _filter_cases(cases: list, patterns: tuple[str, ...]) -> list:
    """Keep the cases whose name matches any pattern (exact or glob, any case)."""
    from fnmatch import fnmatchcase

    if not patterns:
        return cases
    lowered = [p.lower() for p in patterns]
    return [c for c in cases if any(fnmatchcase(c.name.lower(), p) for p in lowered)]


def _truncate(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
