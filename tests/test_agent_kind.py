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


def test_agent_json_output_is_parsed(tmp_path, monkeypatch):
    def fake_model(name, model_cfg):
        return DeterministicModel(outputs=[make_output("done", [{
            "command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo '{\"drc_errors\": 0}'"
        }])])

    monkeypatch.setattr(builtin_kinds, "_agent_model", fake_model)
    _, _, results = run_models({
        "board": f'{{{{ config(model_type="agent", agent_dir="{tmp_path}", output_format="json") }}}}\n'
                 "build it",
        "after": '{{ config(model_type="template") }}errors: {{ ref("board").output.drc_errors }}',
    })
    assert results["board"].value["output"] == {"drc_errors": 0}
    assert results["after"].llm_output.strip() == "errors: 0"


def test_agent_files_become_the_models_files(tmp_path, monkeypatch):
    calls = []

    def fake_model(name, model_cfg):
        calls.append(name)
        return DeterministicModel(outputs=[
            make_output("write", [{"command": (
                "d=$(ls -d .pbt_out_*) && echo '# SS34' > $d/SS34.md "
                "&& mkdir $d/crops && printf png > $d/crops/band.png"
            )}]),
            make_output("done", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo noted"}]),
        ])

    monkeypatch.setattr(builtin_kinds, "_agent_model", fake_model)
    from pbt.storage import MemoryStorageBackend

    storage = MemoryStorageBackend()
    models = {
        "research": f'{{{{ config(model_type="agent", agent_dir="{tmp_path}") }}}}\nresearch SS34',
        "names": (
            '{{ config(model_type="template") }}'
            "{% for f in ref('research').files %}{{ f.name }} {% endfor %}"
        ),
    }
    _, _, results = run_models(models, storage=storage)

    value = results["research"].value
    assert value["output"].strip() == "noted"
    by_name = {f.name: f for f in value["files"]}
    assert by_name["SS34.md"].read_bytes() == b"# SS34\n"
    assert by_name["crops"].files()[0].read_bytes() == b"png"
    assert results["names"].llm_output.split() == ["SS34.md", "crops"]
    assert not list(tmp_path.glob(".pbt_out_*")), "the output directory is removed"

    _, _, again = run_models(models, storage=storage)
    assert len(calls) == 1, "second run is served from cache"
    assert {f.name for f in again["research"].value["files"]} == {"SS34.md", "crops"}
