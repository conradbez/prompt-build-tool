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
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pbt.executor.parser import render_prompt
from pbt.storage.base import StorageBackend


@dataclass
class TestResult:
    test_name: str
    status: str          # 'pass' | 'fail' | 'error'
    prompt_rendered: str = ""
    llm_output: str = ""
    error: str = ""
    execution_ms: int = 0


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


def execute_tests(
    run_id: str,
    tests: dict[str, str],
    model_outputs: dict[str, str],
    storage_backend: StorageBackend,
    on_test_start: Callable[[str], None] | None = None,
    on_test_done: Callable[[TestResult], None] | None = None,
    llm_call: Callable[[str], str] | None = None,
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
    """
    if llm_call is None:
        raise ValueError(
            "llm_call must be provided to execute_tests(). "
            "Use pbt.llm.resolve_llm_call(models_dir) to auto-discover from client.py."
        )

    results: list[TestResult] = []

    for test_name in sorted(tests):
        source = tests[test_name]

        if on_test_start:
            on_test_start(test_name)

        try:
            rendered, _ = render_prompt(source, model_outputs)
            t0 = time.monotonic()
            llm_output = llm_call(rendered)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            passed = _parse_pass(llm_output)
            result = TestResult(
                test_name=test_name,
                status="pass" if passed else "fail",
                prompt_rendered=rendered,
                llm_output=llm_output,
                execution_ms=elapsed_ms,
            )

        except Exception as exc:  # noqa: BLE001
            result = TestResult(
                test_name=test_name,
                status="error",
                error=str(exc),
            )

        storage_backend.record_test_result(run_id, result)
        results.append(result)

        if on_test_done:
            on_test_done(result)

    return results
