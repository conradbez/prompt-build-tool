"""The ``agent`` model kind: mini-swe-agent run in a directory."""

from __future__ import annotations

import pytest

pytest.importorskip("minisweagent")

from minisweagent.models.test_models import DeterministicModel, make_output

from pbt.executor import builtin_kinds
from tests.test_model_types import run_models


@pytest.fixture
def scripted_agent(monkeypatch):
    """Drive the agent with a fixed script of shell commands instead of an LLM."""
    seen = {}

    def fake_model(name, model_cfg):
        seen["name"] = name
        return DeterministicModel(outputs=[
            make_output("write the file", [{"command": "echo hi > note.txt"}]),
            make_output("done", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat note.txt"}]),
        ])

    monkeypatch.setattr(builtin_kinds, "_agent_model", fake_model)
    return seen


def test_agent_runs_in_dir_and_returns_output_logs_time(tmp_path, scripted_agent):
    workdir = tmp_path / "work"
    _, _, results = run_models({
        "fix": f'{{{{ config(model_type="agent", agent_dir="{workdir}", agent_model="m1") }}}}\n'
               "write hi to note.txt",
        "after": '{{ config(model_type="template") }}got: {{ ref("fix")["output"] }}',
    })

    assert (workdir / "note.txt").read_text() == "hi\n"
    assert scripted_agent["name"] == "m1"

    assert set(results["fix"].value) == {"output", "logs", "time_run"}
    assert results["after"].llm_output.strip() == "got: hi"


def test_agent_output_shape(tmp_path, scripted_agent):
    from pbt.model_spec import ModelSpec
    from pbt.model_types import ModelCall
    import asyncio

    spec = ModelSpec(name="a", source="", model_type="agent", config={"agent_dir": str(tmp_path)})

    async def compute(rendered, compute):
        return await compute()

    call = ModelCall(spec=spec, outputs={}, llm=None, compute=compute)
    value = asyncio.run(builtin_kinds.run_agent("task", call))

    assert value["output"] == "hi\n"
    assert isinstance(value["time_run"], float)
    assert value["logs"][0]["role"] == "system"
    assert value["logs"][-1]["role"] == "exit"


def test_agent_requires_agent_dir(scripted_agent):
    _, _, results = run_models({"a": '{{ config(model_type="agent") }}do it'})
    assert "agent_dir" in (results["a"].error or "")
