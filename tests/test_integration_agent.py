"""
Integration test: an ``agent`` model doing a small programming task with a
real Gemini model through mini-swe-agent.

Requires GEMINI_API_KEY (read from the environment or .env); skipped otherwise.
Override the model with PBT_AGENT_TEST_MODEL (a litellm model name).
"""

from __future__ import annotations

import importlib.util
import os

import pytest
from dotenv import load_dotenv

pytest.importorskip("minisweagent")
load_dotenv()

from tests.test_model_types import run_models

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)

MODEL = os.environ.get("PBT_AGENT_TEST_MODEL", "gemini/gemini-3-flash-preview")

TASK = """\
Create a file fizzbuzz.py in the current directory defining a function
fizzbuzz(n: int) -> list[str] that returns the FizzBuzz sequence for 1..n:
"Fizz" for multiples of 3, "Buzz" for multiples of 5, "FizzBuzz" for both,
and the number as a string otherwise. Run it once to check it works.
When done, submit the single word DONE as your final output."""


def test_agent_writes_working_code_with_gemini(tmp_path):
    workdir = tmp_path / "work"
    _, _, results = run_models({
        "write_fizzbuzz": (
            f'{{{{ config(model_type="agent", agent_dir="{workdir}", '
            f'agent_model="{MODEL}", agent_step_limit="15", agent_cost_limit="0") }}}}\n'
            + TASK
        ),
        "report": '{{ config(model_type="template") }}agent said: {{ ref("write_fizzbuzz")["output"] }}',
    })

    result = results["write_fizzbuzz"]
    assert result.status == "success", result.error

    value = result.value
    assert set(value) == {"output", "logs", "time_run"}
    assert "DONE" in value["output"], [(m.get("role"), str(m.get("content"))[:300], m.get("extra", {}).get("exit_status")) for m in value["logs"][-4:]]
    assert value["time_run"] > 0
    assert len(value["logs"]) > 2

    # The code the agent wrote actually works.
    spec = importlib.util.spec_from_file_location("fizzbuzz", workdir / "fizzbuzz.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.fizzbuzz(15) == [
        "1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz",
        "11", "Fizz", "13", "14", "FizzBuzz",
    ]

    # The output reached a downstream model.
    assert "DONE" in results["report"].value
