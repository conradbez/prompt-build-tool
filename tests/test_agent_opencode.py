"""The ``agent`` model kind with ``agent_backend="opencode"``, driven by a fake binary."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path

import pytest

from pbt.executor import builtin_kinds
from pbt.model_spec import ModelSpec
from pbt.model_types import ModelCall
from tests.test_model_types import run_models


def _event(type_, **part):
    return json.dumps({"type": type_, "sessionID": "s1", "part": part})


FAKE_SCRIPT = """#!/bin/sh
# Record how we were called, then behave like `opencode run --format json`.
printf '%s\\n' "$@" > invocation.txt
printf '%s' "${{OPENCODE_CONFIG_CONTENT:-}}" > config_content.txt
echo hi > note.txt
cat <<'JSON'
{events}
JSON
exit {exit_code}
"""


@pytest.fixture
def fake_opencode(tmp_path, monkeypatch):
    """Install a shell script standing in for the opencode binary."""
    def install(events: list[str], exit_code: int = 0) -> Path:
        script = tmp_path / "opencode"
        script.write_text(FAKE_SCRIPT.format(events="\n".join(events), exit_code=exit_code))
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PBT_OPENCODE_BIN", str(script))
        return script
    return install


DONE_EVENTS = [
    _event("step_start", type="step-start", messageID="m1"),
    _event("tool_use", type="tool", tool="bash", messageID="m1"),
    _event("step_finish", type="step-finish", messageID="m1", cost=0.001),
    _event("text", type="text", messageID="m1", text="working..."),
    _event("step_start", type="step-start", messageID="m2"),
    _event("text", type="text", messageID="m2", text="DO"),
    _event("text", type="text", messageID="m2", text="NE"),
    _event("step_finish", type="step-finish", messageID="m2", cost=0.002),
]


def test_opencode_runs_in_dir_and_returns_last_message(tmp_path, fake_opencode):
    fake_opencode(DONE_EVENTS)
    workdir = tmp_path / "work"
    _, _, results = run_models({
        "fix": f'{{{{ config(model_type="agent", agent_backend="opencode", '
               f'agent_dir="{workdir}", agent_model="openai/gpt-4o-mini") }}}}\n'
               "write hi to note.txt",
        "after": '{{ config(model_type="template") }}got: {{ ref("fix")["output"] }}',
    })

    assert results["fix"].status == "success", results["fix"].error
    assert (workdir / "note.txt").read_text() == "hi\n"
    assert results["after"].llm_output.strip() == "got: DONE"

    value = results["fix"].value
    assert set(value) == {"output", "logs", "time_run"}
    assert [e["type"] for e in value["logs"]] == [json.loads(e)["type"] for e in DONE_EVENTS]

    argv = (workdir / "invocation.txt").read_text().splitlines()
    assert argv[:5] == ["run", "--format", "json", "--auto", "--dir"]
    assert "--model" in argv and argv[argv.index("--model") + 1] == "openai/gpt-4o-mini"
    task_arg = (workdir / "invocation.txt").read_text().split("--model\nopenai/gpt-4o-mini\n", 1)[1]
    assert task_arg.lstrip().startswith("write hi to note.txt\n\nYour working directory is ")
    assert str(workdir) in task_arg
    assert (workdir / "config_content.txt").read_text() == ""


def _run_opencode(workdir: Path, **config) -> dict:
    spec = ModelSpec(
        name="a", source="", model_type="agent",
        config={"agent_dir": str(workdir), "agent_backend": "opencode", **config},
    )

    async def compute(rendered, compute):
        return await compute()

    call = ModelCall(spec=spec, outputs={}, llm=None, compute=compute)
    return asyncio.run(builtin_kinds.run_agent("task", call))


def test_opencode_mcp_dict_is_passed_through_config_content(tmp_path, fake_opencode, monkeypatch):
    fake_opencode(DONE_EVENTS)
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising=False)
    servers = {"fs": {"type": "local", "command": ["npx", "-y", "server-fs"], "enabled": True}}

    _run_opencode(tmp_path, agent_mcp=json.dumps(servers))

    assert json.loads((tmp_path / "config_content.txt").read_text()) == {"mcp": servers}


def test_opencode_mcp_from_jinja_dict_literal(tmp_path, fake_opencode, monkeypatch):
    fake_opencode(DONE_EVENTS)
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising=False)
    workdir = tmp_path / "work"
    _, _, results = run_models({
        "a": '{{ config(model_type="agent", agent_backend="opencode", agent_dir="%s", '
             'agent_mcp={"docs": {"type": "remote", "url": "https://mcp.example.com"}}) }}task' % workdir,
    })
    assert results["a"].status == "success", results["a"].error
    assert json.loads((workdir / "config_content.txt").read_text()) == {
        "mcp": {"docs": {"type": "remote", "url": "https://mcp.example.com"}}
    }


def test_opencode_mcp_from_opencode_json_file_and_existing_env(tmp_path, fake_opencode, monkeypatch):
    fake_opencode(DONE_EVENTS)
    cfg = tmp_path / "opencode.json"
    cfg.write_text(json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "mcp": {"fs": {"type": "local", "command": ["x"]}},
    }))
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps({"model": "openai/gpt-4o", "mcp": {"old": {"type": "remote", "url": "u"}}}))

    _run_opencode(tmp_path, agent_mcp=str(cfg))

    assert json.loads((tmp_path / "config_content.txt").read_text()) == {
        "model": "openai/gpt-4o",
        "mcp": {"old": {"type": "remote", "url": "u"}, "fs": {"type": "local", "command": ["x"]}},
    }


def test_opencode_step_limit_stops_the_run(tmp_path, fake_opencode):
    fake_opencode(DONE_EVENTS)
    value = _run_opencode(tmp_path, agent_step_limit="1")
    assert value["logs"][-1] == {"type": "exit", "reason": "step limit (1) reached"}
    assert value["output"] == ""   # stopped before the final message


def test_opencode_cost_limit_stops_the_run(tmp_path, fake_opencode):
    fake_opencode(DONE_EVENTS)
    value = _run_opencode(tmp_path, agent_cost_limit="0.0005")
    assert value["logs"][-1]["reason"].startswith("cost limit")


def test_opencode_failure_is_reported(tmp_path, fake_opencode):
    fake_opencode([_event("error", error="boom")], exit_code=2)
    with pytest.raises(RuntimeError, match="exited with 2.*boom"):
        _run_opencode(tmp_path)


def test_opencode_missing_binary(tmp_path, monkeypatch):
    monkeypatch.setenv("PBT_OPENCODE_BIN", "")
    monkeypatch.setattr(builtin_kinds.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="npm i -g opencode-ai"):
        _run_opencode(tmp_path)


def test_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="agent_backend must be one of"):
        _run_opencode(tmp_path, agent_backend="codex")
