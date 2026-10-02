"""
Integration test: an ``agent`` model with ``agent_backend="opencode"`` doing a
small programming task through a real opencode binary and OpenAI model.

Requires the ``opencode`` binary (PATH or PBT_OPENCODE_BIN) and one provider
key (environment or .env): OPENAI_API_KEY, ANTHROPIC_API_KEY or GEMINI_API_KEY;
skipped otherwise.  Override the model with PBT_OPENCODE_TEST_MODEL (an
opencode ``provider/model`` name).
"""

from __future__ import annotations

import importlib.util
import os

import pytest
from dotenv import load_dotenv

from pbt.executor.builtin_kinds import _opencode_bin
from tests.test_model_types import run_models

load_dotenv()

#: A cheap tool-capable model per provider key, first one present wins.
_MODEL_FOR_KEY = {
    "OPENAI_API_KEY": "openai/gpt-4.1-mini",
    "ANTHROPIC_API_KEY": "anthropic/claude-haiku-4-5",
    "GEMINI_API_KEY": "google/gemini-3.8-flash",
}
DEFAULT_MODEL = next((m for k, m in _MODEL_FOR_KEY.items() if os.environ.get(k)), None)
MODEL = os.environ.get("PBT_OPENCODE_TEST_MODEL") or DEFAULT_MODEL

pytestmark = pytest.mark.skipif(
    not (_opencode_bin() and MODEL),
    reason="opencode binary or a provider key (OPENAI/ANTHROPIC/GEMINI_API_KEY) not available",
)

TASK = """\
Create a file fizzbuzz.py in the current directory defining a function
fizzbuzz(n: int) -> list[str] that returns the FizzBuzz sequence for 1..n:
"Fizz" for multiples of 3, "Buzz" for multiples of 5, "FizzBuzz" for both,
and the number as a string otherwise. Run it once to check it works.
When done, reply with the single word DONE."""


def test_opencode_writes_working_code(tmp_path):
    workdir = tmp_path / "work"
    _, _, results = run_models({
        "write_fizzbuzz": (
            f'{{{{ config(model_type="agent", agent_backend="opencode", agent_dir="{workdir}", '
            f'agent_model="{MODEL}", agent_step_limit="15", agent_cost_limit="0.5") }}}}\n'
            + TASK
        ),
        "report": '{{ config(model_type="template") }}agent said: {{ ref("write_fizzbuzz")["output"] }}',
    })

    result = results["write_fizzbuzz"]
    assert result.status == "success", result.error

    value = result.value
    assert set(value) == {"output", "logs", "time_run"}
    trail = [
        (e.get("type"), (e.get("part") or {}).get("reason") or (e.get("part") or {}).get("tool") or e.get("error"))
        for e in value["logs"][-8:]
    ]
    assert "DONE" in value["output"], (value["output"][:500], trail)
    assert value["time_run"] > 0
    assert any(e.get("type") == "tool_use" for e in value["logs"])

    tool_calls = [
        (e["part"].get("tool"), e["part"].get("state", {}).get("input"))
        for e in value["logs"] if e.get("type") == "tool_use"
    ]
    assert (workdir / "fizzbuzz.py").exists(), tool_calls
    spec = importlib.util.spec_from_file_location("fizzbuzz", workdir / "fizzbuzz.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.fizzbuzz(15) == [
        "1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz",
        "11", "Fizz", "13", "14", "FizzBuzz",
    ]
