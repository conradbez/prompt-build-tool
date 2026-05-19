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

Parameterised tests via inline config()
----------------------------------------
Each test file can declare its own promptdata and promptfiles via a
``{{ config() }}`` call at the top of the file::

    {{ config(
        promptdata={"topic": "AI"},
        promptfiles={"instructions": "path/to/file.txt"}
    ) }}

When the value is a list, the test runs once per entry (zipped, not
cross-joined).  All test files in the same suite must declare the same
number of rows when using lists.

    {{ config(
        promptdata=[{"topic": "AI"}, {"topic": "ML"}],
        promptfiles=[{"instructions": "ai.txt"}, {"instructions": "ml.txt"}]
    ) }}
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pbt.executor.parser_initial import extract_test_config
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
    param_label: str = ""   # e.g. "row_1" when using parameterised rows


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


def get_test_row_count(tests: dict[str, str]) -> int:
    """
    Determine the number of parameterised rows declared across all test files.

    Returns 0 if no test declares a list-valued ``promptdata`` config.
    Raises ``ValueError`` if tests declare inconsistent row counts.
    """
    row_count = 0
    for name, source in tests.items():
        cfg = extract_test_config(source)
        pd = cfg.get("promptdata")
        pf = cfg.get("promptfiles")

        # Determine this test's row count from list-typed values
        test_rows = 0
        if isinstance(pd, list):
            test_rows = len(pd)
        if isinstance(pf, list):
            if test_rows == 0:
                test_rows = len(pf)
            elif test_rows != len(pf):
                raise ValueError(
                    f"Test '{name}': promptdata and promptfiles lists must have "
                    f"the same length (got {len(pd)} vs {len(pf)})."
                )

        if test_rows == 0:
            continue

        if row_count == 0:
            row_count = test_rows
        elif row_count != test_rows:
            raise ValueError(
                f"Test '{name}' declares {test_rows} row(s) but other tests "
                f"declare {row_count}. All parameterised tests must have the "
                f"same number of rows."
            )

    return row_count


def get_merged_row_params(
    tests: dict[str, str],
    row_index: int,
) -> tuple[dict, dict]:
    """
    Merge promptdata and promptfiles from all test configs for a given row index.

    Returns ``(merged_promptdata, merged_promptfiles)``.
    Used by the CLI to determine what parameters to pass to the model run.
    """
    merged_pd: dict = {}
    merged_pf: dict = {}

    for source in tests.values():
        cfg = extract_test_config(source)
        pd = cfg.get("promptdata")
        pf = cfg.get("promptfiles")

        if isinstance(pd, list):
            if row_index < len(pd) and isinstance(pd[row_index], dict):
                merged_pd.update(pd[row_index])
        elif isinstance(pd, dict):
            merged_pd.update(pd)

        if isinstance(pf, list):
            if row_index < len(pf) and isinstance(pf[row_index], dict):
                merged_pf.update(pf[row_index])
        elif isinstance(pf, dict):
            merged_pf.update(pf)

    return merged_pd, merged_pf


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


def _resolve_row_params(
    cfg: dict,
    row_index: int | None,
) -> tuple[dict | None, dict | None]:
    """
    Extract promptdata and promptfiles for a single test execution.

    When *row_index* is None, single-dict configs are used as-is.
    When *row_index* is an int, list configs are indexed; single-dict configs
    are used as-is for every row.
    """
    pd_val = cfg.get("promptdata")
    pf_val = cfg.get("promptfiles")

    if row_index is not None and isinstance(pd_val, list):
        promptdata = pd_val[row_index] if row_index < len(pd_val) else None
    elif isinstance(pd_val, dict):
        promptdata = pd_val
    else:
        promptdata = None

    if row_index is not None and isinstance(pf_val, list):
        promptfiles = pf_val[row_index] if row_index < len(pf_val) else None
    elif isinstance(pf_val, dict):
        promptfiles = pf_val
    else:
        promptfiles = None

    return promptdata or None, promptfiles or None


def execute_tests(
    run_id: str,
    tests: dict[str, str],
    model_outputs: dict[str, str],
    storage_backend: StorageBackend,
    on_test_start: Callable[[str], None] | None = None,
    on_test_done: Callable[[TestResult], None] | None = None,
    llm_call: Callable[[str], str] | None = None,
    row_index: int | None = None,
) -> list[TestResult]:
    """
    Execute each test prompt against the given model outputs.

    Parameters
    ----------
    run_id:
        The run whose model outputs are being tested.
    tests:
        Mapping of test_name → raw prompt source, from load_tests().
    model_outputs:
        Mapping of model_name → LLM output, used to resolve ref() calls.
    llm_call:
        LLM backend callable ``(prompt: str) -> str``. Required.
    row_index:
        When not None, selects which entry from list-valued ``promptdata``
        and ``promptfiles`` configs to use for this execution.  Test names
        are suffixed with ``[row_N]`` (1-indexed).
        When None, single-dict configs are used as-is and no suffix is added.
    """
    if llm_call is None:
        raise ValueError(
            "llm_call must be provided to execute_tests(). "
            "Use pbt.llm.resolve_llm_call(models_dir) to auto-discover from client.py."
        )

    work: list[tuple[str, str, dict | None, dict | None]] = []

    for test_name in sorted(tests):
        source = tests[test_name]
        cfg = extract_test_config(source)
        promptdata, promptfiles = _resolve_row_params(cfg, row_index)

        if row_index is not None:
            display_name = f"{test_name}[row_{row_index + 1}]"
        else:
            display_name = test_name

        work.append((display_name, source, promptdata, promptfiles))

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
