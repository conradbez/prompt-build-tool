"""
pbt test runner — discovers and executes *.prompt test files.

Test files live in the tests/ directory (sibling to models/).
They have full Jinja2 context (ref() works just like in model prompts).

Pass / fail rule
----------------
The LLM response must be valid JSON containing ``"results": "pass"``.
Any other response — wrong JSON, extra fields, wrong value — is a failure.

Example test (tests/smoke_test.prompt):
    Always respond with exactly this JSON: {"results": "pass"}

Example test that inspects a model output (tests/haiku_has_lines.prompt):
    The following haiku should have exactly 3 lines:

    {{ ref('haiku') }}

    If it has 3 lines respond {"results": "pass"}, otherwise {"results": "fail"}.

Parameterised tests via promptparams.csv
-----------------------------------------
When a ``promptparams.csv`` file is present (or supplied explicitly), every
test is *cross-joined* with each row in the file.  A test named ``smoke``
with 3 parameter rows becomes ``smoke[row_1]``, ``smoke[row_2]``,
``smoke[row_3]``.

See ``pbt.promptparams`` for the CSV column-naming convention.
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pbt.executor.parser_model import render_prompt
from pbt.storage.base import StorageBackend


@dataclass
class TestResult:
    test_name: str
    status: str          # 'pass' | 'fail' | 'error'
    prompt_rendered: str = ""
    llm_output: str = ""
    error: str = ""
    execution_ms: int = 0
    param_label: str = ""   # e.g. "row_1" when using promptparams rows


def load_tests(tests_dir: str | Path = "tests") -> dict[str, str]:
    """
    Discover *.prompt files in *tests_dir*.

    Returns a mapping of test_name → raw source.
    Returns an empty dict (not an error) when the directory doesn't exist,
    so `pbt test` gives a friendly message rather than crashing.
    """
    tests_dir = Path(tests_dir)
    if not tests_dir.exists():
        return {}
    return {
        f.stem: f.read_text(encoding="utf-8")
        for f in sorted(tests_dir.glob("*.prompt"))
    }


def _parse_pass(llm_output: str) -> bool:
    """
    Return True iff *llm_output* is (or contains) JSON with ``results == "pass"``.

    Handles optional markdown code fences (```json … ```) that some LLMs add.
    """
    text = llm_output.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (``` or ```json) and last line (```)
        inner = lines[1:-1] if len(lines) > 2 else lines
        text = "\n".join(inner).strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False

    return isinstance(data, dict) and data.get("results") == "pass"


def _invoke_llm(
    rendered: str,
    llm_call: Callable,
    promptfiles: dict[str, str | list[str]] | None = None,
) -> str:
    """
    Call *llm_call* with *rendered*, optionally passing opened file objects
    for each entry in *promptfiles* when the callable accepts a ``files``
    parameter.

    Files are opened in binary mode and it is the caller's responsibility
    that paths exist.  We open them here so that the ``llm_call`` consumer
    receives ready-to-read file objects, mirroring the behaviour in
    ``execute_run``.
    """
    _sig = inspect.signature(llm_call).parameters
    _kwargs: dict = {}

    if promptfiles and "files" in _sig:
        open_files = []
        for path_or_list in promptfiles.values():
            if isinstance(path_or_list, list):
                for p in path_or_list:
                    open_files.append(open(p, "rb"))  # noqa: WPS515
            else:
                open_files.append(open(path_or_list, "rb"))  # noqa: WPS515
        _kwargs["files"] = open_files

    return llm_call(rendered, **_kwargs)


def execute_tests(
    run_id: str,
    tests: dict[str, str],
    model_outputs: dict[str, str],
    storage_backend: StorageBackend,
    on_test_start: Callable[[str], None] | None = None,
    on_test_done: Callable[[TestResult], None] | None = None,
    llm_call: Callable[[str], str] | None = None,
    promptparams_rows: list[dict[str, str]] | None = None,
) -> list[TestResult]:
    """
    Execute each test prompt against the given model outputs.

    Parameters
    ----------
    run_id:
        The run whose model outputs are being tested.  Test results are
        stored in test_results linked to this run_id.
    tests:
        Mapping of test_name → raw prompt source, from load_tests().
    model_outputs:
        Mapping of model_name → LLM output, used to resolve ref() calls.
    llm_call:
        LLM backend callable ``(prompt: str) -> str``. Required.
    promptparams_rows:
        Optional list of raw CSV row dicts from ``promptparams.load_promptparams()``.
        When supplied, every test is cross-joined with every row so that
        ``len(tests) × len(promptparams_rows)`` test cases are executed.
        Each test name is suffixed with ``[row_N]`` (1-indexed).
        When *None* or empty, tests run once with no extra parameters.
    """
    if llm_call is None:
        raise ValueError(
            "llm_call must be provided to execute_tests(). "
            "Use pbt.llm.resolve_llm_call(models_dir) to auto-discover from client.py."
        )

    from pbt.promptparams import parse_promptparams_row

    # Build the list of (test_name_display, source, promptdata, promptfiles) to run.
    # Without rows: one entry per test, no params.
    # With rows: cross-join tests × rows.
    work: list[tuple[str, str, dict | None, dict | None]] = []

    if promptparams_rows:
        for test_name in sorted(tests):
            source = tests[test_name]
            for idx, row in enumerate(promptparams_rows, start=1):
                label = f"row_{idx}"
                display_name = f"{test_name}[{label}]"
                promptdata, promptfiles = parse_promptparams_row(row)
                work.append((
                    display_name,
                    source,
                    promptdata or None,
                    promptfiles or None,
                ))
    else:
        for test_name in sorted(tests):
            work.append((test_name, tests[test_name], None, None))

    results: list[TestResult] = []

    for display_name, source, promptdata, promptfiles in work:
        if on_test_start:
            on_test_start(display_name)

        try:
            rendered, _ = render_prompt(
                source,
                model_outputs,
                promptdata=promptdata,
                model_name=display_name,
            )
            cached = storage_backend.get_cached_llm_output(rendered)
            if cached is not None:
                llm_output = cached
                elapsed_ms = 0
            else:
                t0 = time.monotonic()
                llm_output = _invoke_llm(rendered, llm_call, promptfiles)
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                storage_backend.mark_model_success(run_id, display_name, rendered, llm_output, cache_key=rendered)

            passed = _parse_pass(llm_output)
            # Extract param_label from display_name if present
            param_label = ""
            if "[" in display_name:
                param_label = display_name.split("[", 1)[1].rstrip("]")

            result = TestResult(
                test_name=display_name,
                status="pass" if passed else "fail",
                prompt_rendered=rendered,
                llm_output=llm_output,
                execution_ms=elapsed_ms,
                param_label=param_label,
            )

        except Exception as exc:  # noqa: BLE001
            result = TestResult(
                test_name=display_name,
                status="error",
                error=str(exc),
            )

        storage_backend.record_test_result(run_id, result)
        results.append(result)

        if on_test_done:
            on_test_done(result)

    return results
