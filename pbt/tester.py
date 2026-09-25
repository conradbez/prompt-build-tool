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

Parameterised tests via promptparams
------------------------------------
Given named cases (``pbt.promptparams.load_cases()``), every test is
*cross-joined* with each case.  A test named ``smoke`` with cases ``Formal``
and ``Casual tone`` becomes ``smoke[Formal]`` and ``smoke[Casual tone]``.

See ``pbt.promptparams`` for the YAML format, baselines and inheritance.

Attaching model files
---------------------
Tests use the same ``config(promptfiles=[...])`` syntax as models.  Naming a
model attaches the files it produced (decoded from their stored ``$pbt``
references), so a judge can look at the actual bytes rather than a handle::

    {{ config(promptfiles=["logo"]) }}
    Is the attached image a fox? Respond {"results": "pass"} or {"results": "fail"}.

Global instructions
-------------------
Test prompts deliberately do **not** receive the run's global instruction (see
``pbt.global_instruction``).  The models under test render with it exactly as
they do in a real run, but a judge told how to write is a biased judge.
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pbt.executor.parser_model import render_prompt
from pbt.promptparams import TestCase
from pbt.storage.base import StorageBackend


@dataclass
class TestResult:
    test_name: str
    status: str          # 'pass' | 'fail' | 'error'
    prompt_rendered: str = ""
    llm_output: str = ""
    error: str = ""
    execution_ms: int = 0
    param_label: str = ""   # the promptparams case name, when there is one


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


def _open_test_files(
    source: str,
    model_outputs: dict,
    promptfiles: dict[str, str | list[str]] | None,
) -> list | None:
    """
    Open the files a test attaches to its LLM call, or return None.

    A test that declares ``{{ config(promptfiles=[...]) }}`` gets exactly those,
    resolved the way a model's are: a name that names a model attaches the
    files that model produced (``"logo"`` for all of them, ``"logo.image"`` for
    one), any other name is a run-level promptfile.  A test that declares none
    keeps the old behaviour and receives every run-level promptfile.

    Run-level files are opened in binary mode; it is the caller's
    responsibility that paths exist.
    """
    from pbt.executor.graph import _parse_promptfiles
    from pbt.executor.parser_initial import parse_model_config
    from pbt.files import iter_files, select_path, split_model_path

    def _open_run_level(path_or_list) -> list:
        paths = path_or_list if isinstance(path_or_list, list) else [path_or_list]
        return [open(p, "rb") for p in paths]  # noqa: WPS515

    declared = _parse_promptfiles(parse_model_config(source))
    if not declared:
        if not promptfiles:
            return None
        return [f for value in promptfiles.values() for f in _open_run_level(value)]

    opened: list = []
    for name in declared:
        upstream = split_model_path(name, model_outputs)
        if upstream is not None:
            model, path = upstream
            value = select_path(model_outputs[model], path, name)
            found = [file for _, file in iter_files(value)]
            if not found:
                raise ValueError(
                    f"Test attaches promptfile '{name}', but model '{model}' "
                    "produced no files there."
                )
            opened.extend(file.open() for file in found)
            continue

        if not promptfiles or name not in promptfiles:
            raise ValueError(
                f"Test declares promptfile '{name}' in config but it is neither "
                f"a model nor a provided promptfile. Pass it via --promptfile {name}=path."
            )
        opened.extend(_open_run_level(promptfiles[name]))

    return opened or None


def _invoke_llm(rendered: str, llm_call: Callable, files: list | None = None) -> str:
    """
    Call *llm_call* with *rendered*, passing *files* when the callable
    accepts a ``files`` parameter — the same contract as model calls.
    """
    if files and "files" in inspect.signature(llm_call).parameters:
        return llm_call(rendered, files=files)
    return llm_call(rendered)


def execute_tests(
    run_id: str,
    tests: dict[str, str],
    model_outputs: dict[str, str],
    storage_backend: StorageBackend,
    on_test_start: Callable[[str], None] | None = None,
    on_test_done: Callable[[TestResult], None] | None = None,
    llm_call: Callable[[str], str] | None = None,
    cases: list[TestCase] | None = None,
    promptdata: dict | None = None,
    promptfiles: dict[str, str | list[str]] | None = None,
    param_label: str = "",
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
    cases:
        Optional list of :class:`~pbt.promptparams.TestCase` from
        ``promptparams.load_cases()``.  When supplied, every test is
        cross-joined with every case so that ``len(tests) × len(cases)`` test
        cases are executed, each named ``test[case name]``.
        When *None* or empty, tests run once with no extra parameters.
    promptdata, promptfiles:
        Optional params injected into every test template for a single
        invocation.  Used by the CLI's per-case mode, which runs the models
        once per case and then calls ``execute_tests`` for that case.  Ignored
        when *cases* is supplied.
    param_label:
        Optional label (the case name) that suffixes every test name as
        ``name[label]``, so per-case CLI results stay identifiable.  Ignored
        when *cases* is supplied.
    """
    if llm_call is None:
        raise ValueError(
            "llm_call must be provided to execute_tests(). "
            "Use pbt.llm.resolve_llm_call(models_dir) to auto-discover from client.py."
        )

    from pbt.executor.run_context import _files_hash
    from pbt.files import FileOutputError, blob_store_for, decode_output

    # Stored outputs of file-producing models decode to File/Dir/Output
    # objects, so ref() in a test renders their handle, not the raw envelope.
    blobs = blob_store_for(storage_backend)

    def _decoded(raw):
        if not isinstance(raw, str):
            return raw
        try:
            return decode_output(raw, blobs)
        except FileOutputError:
            return raw

    model_outputs = {name: _decoded(raw) for name, raw in model_outputs.items()}

    # Build the list of (display_name, label, source, promptdata, promptfiles).
    # Without cases: one entry per test.  With cases: cross-join tests × cases.
    work: list[tuple[str, str, str, dict | None, dict | None]] = []

    if cases:
        for test_name in sorted(tests):
            for case in cases:
                work.append((
                    f"{test_name}[{case.name}]",
                    case.name,
                    tests[test_name],
                    case.promptdata or None,
                    case.promptfiles or None,
                ))
    else:
        for test_name in sorted(tests):
            display_name = f"{test_name}[{param_label}]" if param_label else test_name
            work.append((display_name, param_label, tests[test_name], promptdata, promptfiles))

    results: list[TestResult] = []

    for display_name, label, source, promptdata, promptfiles in work:
        if on_test_start:
            on_test_start(display_name)

        try:
            rendered, _ = render_prompt(
                source,
                model_outputs,
                promptdata=promptdata,
                model_name=display_name,
            )
            files = _open_test_files(source, model_outputs, promptfiles)
            # Attached bytes are part of the cache key, so a changed file
            # means a fresh verdict rather than a stale one.
            cache_key = rendered + ("\x00" + _files_hash(files) if files else "")
            cached = storage_backend.get_cached_llm_output(cache_key)
            if cached is not None:
                llm_output = cached
                elapsed_ms = 0
            else:
                t0 = time.monotonic()
                llm_output = _invoke_llm(rendered, llm_call, files)
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                storage_backend.mark_model_success(run_id, display_name, rendered, llm_output, cache_key=cache_key)

            passed = _parse_pass(llm_output)
            result = TestResult(
                test_name=display_name,
                status="pass" if passed else "fail",
                prompt_rendered=rendered,
                llm_output=llm_output,
                execution_ms=elapsed_ms,
                param_label=label,
            )

        except Exception as exc:  # noqa: BLE001
            result = TestResult(
                test_name=display_name,
                status="error",
                error=str(exc),
                param_label=label,
            )

        storage_backend.record_test_result(run_id, result)
        results.append(result)

        if on_test_done:
            on_test_done(result)

    return results
