"""
The model kinds pbt ships with.

Each is a :class:`~pbt.model_types.ModelKind` record: some data, and at most one
function that turns a rendered prompt into a value.  Everything around that —
rendering, the prompt cache, JSON parsing, validation, storage, skip
propagation — belongs to the executor and is identical for every kind,
including the ones you register yourself.

``""`` (plain LLM)   send the rendered prompt to the backend
``template``         the rendered text *is* the output (``exec_fn=None``)
``execute_python``   run the rendered text as Python
``agent``            hand the rendered text to a coding agent as its task
                     (mini-swe-agent, or opencode with MCP servers)

Importing this module registers them; :mod:`pbt` does that on import.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from pbt.files import Dir, File, FileOutputError, Output, contains_files
from pbt.model_types import ModelCall, ModelKind, register_model_kind


# ---------------------------------------------------------------------------
# exec_fns
# ---------------------------------------------------------------------------

async def call_the_llm(rendered: str, call: ModelCall) -> Any:
    """The default: send the rendered prompt to the LLM backend."""
    return await call.llm(rendered)


async def run_python(rendered: str, call: ModelCall) -> Any:
    """Run the rendered template as Python source and return what it produced.

    Output is whatever the code prints; failing that, a variable named
    ``output``.  Upstream outputs are available as ``ref('name')`` and in the
    ``model_outputs`` dict.  The code runs in-process with full builtins — a
    .prompt file using this kind is trusted code.

    To produce files, either set ``output`` to a ``File``/``Dir``/``Output``
    (or a dict holding them), or write into ``out_dir`` — an empty directory
    whose contents become the output's files, with anything printed as its
    text.  ``File``, ``Dir`` and ``Output`` are in scope without an import.

    It goes through the same cache as an LLM call, so unchanged code does not
    re-execute on the next run.
    """
    return await call.compute(rendered, compute=lambda: _exec_python(rendered, call))


def _exec_python(rendered: str, call: ModelCall) -> Any:
    with tempfile.TemporaryDirectory(prefix="pbt-out-") as tmp:
        out_dir = Path(tmp)
        namespace: dict = {
            "model_outputs": call.outputs,
            "ref": lambda name: call.outputs.get(name),
            "out_dir": out_dir,
            "File": File,
            "Dir": Dir,
            "Output": Output,
        }
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exec(compile(rendered, f"<{call.spec.name}>", "exec"), namespace)  # noqa: S102

        # Read what was written before the directory is removed.
        written = _collect_out_dir(out_dir)

    value = namespace.get("output")
    if contains_files(value):
        if written:
            raise FileOutputError(
                f"Model '{call.spec.name}' both wrote files to out_dir and set "
                "`output` to files — use one or the other."
            )
        return value

    printed = stdout.getvalue()
    if printed:
        text = printed.rstrip("\n")
    elif "output" in namespace:
        text = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    else:
        text = ""
    return Output(text, files=written) if written else text


def _collect_out_dir(out_dir: Path) -> list:
    """The files and directories left in *out_dir*, as File/Dir objects."""
    items = []
    for path in sorted(out_dir.iterdir()):
        if path.is_symlink():
            raise FileOutputError(f"out_dir: refusing symlink '{path.name}'.")
        items.append(Dir(path) if path.is_dir() else File(path))
    return items


async def run_agent(rendered: str, call: ModelCall) -> dict:
    """Run a coding agent with the rendered template as its task.

    The agent works in ``agent_dir`` (created if missing), running shell
    commands there until it submits.  The output is a dict::

        {"output": <what the agent submitted>,
         "logs": <the full message trajectory>,
         "time_run": <seconds the agent ran>}

    so a downstream model reads ``ref('fix')['output']``.  With
    ``output_format="json"`` the submitted text is parsed, so a downstream
    model reads ``ref('fix').output.key``.

    Config keys:

    ``agent_dir``         working directory (required)
    ``agent_backend``     ``mini-swe-agent`` (default) or ``opencode``
    ``agent_model``       model name; defaults to ``MSWEA_MODEL_NAME`` for
                          mini-swe-agent, to opencode's own config otherwise
    ``agent_step_limit``  max LLM calls, 0 for none (default 0)
    ``agent_cost_limit``  max spend in dollars, 0 for none (default 3)
    ``agent_mcp``         opencode only: MCP servers the agent may use, as a
                          dict (opencode's ``mcp`` block) or a path to an
                          ``opencode.json``

    mini-swe-agent needs ``pip install mini-swe-agent``; opencode needs the
    ``opencode`` binary on PATH (``npm i -g opencode-ai``) or
    ``PBT_OPENCODE_BIN``.  Results are cached on the rendered prompt like any
    other kind, so an unchanged task does not re-run.
    """
    raw = await call.compute(
        rendered, compute=lambda: asyncio.to_thread(_exec_agent, rendered, call)
    )
    result = json.loads(raw)
    if call.spec.output_format == "json":
        from pbt.executor.run_context import parse_json_output

        result["output"] = parse_json_output(result["output"])
    return result


_AGENT_BACKENDS = ("mini-swe-agent", "opencode")


def _exec_agent(task: str, call: ModelCall) -> str:
    config = call.spec.config
    workdir = config.get("agent_dir")
    if not workdir:
        raise ValueError(
            f"Model '{call.spec.name}': model_type=\"agent\" needs "
            "config(agent_dir=\"...\") — the directory the agent works in."
        )
    workdir = Path(workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    backend = str(config.get("agent_backend") or _AGENT_BACKENDS[0])
    if backend not in _AGENT_BACKENDS:
        raise ValueError(
            f"Model '{call.spec.name}': agent_backend must be one of "
            f"{', '.join(_AGENT_BACKENDS)}, not '{backend}'."
        )

    started = time.monotonic()
    if backend == "opencode":
        output, logs = _exec_opencode(task, call, workdir)
    else:
        output, logs = _exec_mini_swe_agent(task, call, workdir)
    time_run = round(time.monotonic() - started, 3)

    return json.dumps({"output": output, "logs": logs, "time_run": time_run}, default=str)


# ---------------------------------------------------------------------------
# Backend: mini-swe-agent
# ---------------------------------------------------------------------------

def _exec_mini_swe_agent(task: str, call: ModelCall, workdir: Path) -> tuple[str, list]:
    try:
        import yaml
        from minisweagent import package_dir
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
    except ImportError as exc:
        raise ImportError(
            "model_type=\"agent\" needs mini-swe-agent: pip install mini-swe-agent"
        ) from exc

    config = call.spec.config
    defaults = yaml.safe_load((package_dir / "config" / "default.yaml").read_text())
    agent_cfg = defaults["agent"] | {
        "step_limit": call.spec.config_int("agent_step_limit", 0),
        "cost_limit": float(config.get("agent_cost_limit", 3.0)),
    }
    agent = DefaultAgent(
        _agent_model(config.get("agent_model") or os.getenv("MSWEA_MODEL_NAME"), defaults["model"]),
        LocalEnvironment(cwd=str(workdir), **defaults["environment"]),
        **agent_cfg,
    )
    result = agent.run(task + _SUBMIT_INSTRUCTION)
    return result.get("submission", ""), agent.messages


#: mini-swe-agent's own prompt says to submit with a bare
#: ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``, which submits nothing.  pbt
#: wants a final output, which is whatever the agent prints after that marker.
_SUBMIT_INSTRUCTION = """

When you are done, submit your final output (the answer to pass on to the next
step) by printing COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT on the first line and
the output after it, e.g.:

    printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n%s\\n' "your final output"
"""


# ---------------------------------------------------------------------------
# Backend: opencode (https://opencode.ai) — brings MCP servers
# ---------------------------------------------------------------------------

def _exec_opencode(task: str, call: ModelCall, workdir: Path) -> tuple[str, list]:
    """Run ``opencode run`` headless in *workdir* and collect its JSON events.

    opencode's final reply (the text parts of its last message) is the output;
    every event it printed is the log.  ``agent_mcp`` is handed over through
    ``OPENCODE_CONFIG_CONTENT``, which opencode merges over its own config, so
    the project's ``opencode.json`` still applies.  Step and cost limits are
    enforced by watching ``step_finish`` events and stopping the process.
    """
    binary = _opencode_bin()
    if not binary:
        raise FileNotFoundError(
            "agent_backend=\"opencode\" needs the opencode binary: "
            "npm i -g opencode-ai (or set PBT_OPENCODE_BIN)"
        )

    config = call.spec.config
    cmd = [binary, "run", "--format", "json", "--auto", "--dir", str(workdir)]
    if config.get("agent_model"):
        cmd += ["--model", str(config["agent_model"])]
    cmd.append(task + _OPENCODE_INSTRUCTION.format(workdir=workdir))

    env = os.environ.copy()
    mcp = _opencode_mcp_servers(config.get("agent_mcp"), call.spec.path)
    if mcp:
        env["OPENCODE_CONFIG_CONTENT"] = _merged_opencode_config(env.get("OPENCODE_CONFIG_CONTENT"), mcp)

    step_limit = call.spec.config_int("agent_step_limit", 0)
    cost_limit = float(config.get("agent_cost_limit", 3.0))

    proc = subprocess.Popen(
        cmd, cwd=str(workdir), env=env, text=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    events: list = []
    steps, cost, stopped = 0, 0.0, None
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            events.append({"type": "raw", "text": line})
            continue
        events.append(event)
        if event.get("type") == "step_finish":
            steps += 1
            cost += float((event.get("part") or {}).get("cost") or 0)
            if step_limit and steps >= step_limit:
                stopped = f"step limit ({step_limit}) reached"
            elif cost_limit and cost >= cost_limit:
                stopped = f"cost limit (${cost_limit}) reached"
            if stopped:
                proc.terminate()
                break
    stderr = proc.stderr.read() if proc.stderr else ""
    proc.wait()

    if stopped:
        events.append({"type": "exit", "reason": stopped})
    elif proc.returncode != 0:
        raise RuntimeError(
            f"opencode exited with {proc.returncode}: {stderr.strip() or _last_error(events)}"
        )

    return _opencode_final_text(events), events


#: Smaller models tend to guess a path for new files; pin them to agent_dir.
_OPENCODE_INSTRUCTION = """

Your working directory is {workdir}. Create and edit files there (relative
paths resolve against it). When you are done, your final reply is the output
passed on to the next step.
"""


def _opencode_bin() -> str | None:
    """Path to the opencode binary — a seam for tests."""
    return os.getenv("PBT_OPENCODE_BIN") or shutil.which("opencode")


def _opencode_mcp_servers(value: Any, model_path: Path) -> dict:
    """``agent_mcp`` as opencode's ``mcp`` dict: a dict, JSON text, or a path."""
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text.startswith("{"):
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (model_path.parent / path) if model_path.name != "<inline>" else path
        text = path.read_text()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("agent_mcp must be a JSON object of MCP servers")
    # A whole opencode.json is fine too: take its "mcp" block.
    if "mcp" in data and all(not isinstance(v, dict) or "type" not in v for v in data.values()):
        return data["mcp"]
    return data


def _merged_opencode_config(existing: str | None, mcp: dict) -> str:
    base: dict = {}
    if existing:
        try:
            base = json.loads(existing)
        except json.JSONDecodeError:
            base = {}
    base["mcp"] = {**base.get("mcp", {}), **mcp}
    return json.dumps(base)


def _opencode_final_text(events: list) -> str:
    """The text parts of the last message opencode wrote."""
    texts = [e["part"] for e in events if e.get("type") == "text" and isinstance(e.get("part"), dict)]
    if not texts:
        return ""
    last_message = texts[-1].get("messageID")
    return "".join(p.get("text", "") for p in texts if p.get("messageID") == last_message)


def _last_error(events: list) -> str:
    for event in reversed(events):
        if event.get("type") == "error":
            return str(event.get("error") or (event.get("part") or {}).get("error"))
    return "no error reported"


def _agent_model(name: str | None, model_cfg: dict):
    """The mini-swe-agent model to drive the agent — a seam for tests."""
    from minisweagent.models import get_model

    return get_model(name, model_cfg)


# ---------------------------------------------------------------------------
# The kinds
# ---------------------------------------------------------------------------

#: The default: render the template, send it to the LLM.
LLM = ModelKind(name="", exec_fn=call_the_llm)

#: Render the template and use it as the output — no LLM call.  For nodes that
#: only reshape what upstream models already produced: a header, a merge of two
#: outputs, a pass-through.  A global instruction is never applied, since the
#: rendered text is the output itself rather than a prompt to answer.
TEMPLATE = ModelKind(name="template", exec_fn=None, accepts_global_instruction=False)

PYTHON = ModelKind(
    name="execute_python",
    exec_fn=run_python,
    accepts_global_instruction=False,
)

#: A coding agent working in a directory, with the rendered text as its task.
AGENT = ModelKind(
    name="agent",
    exec_fn=run_agent,
    config_keys=frozenset({
        "agent_dir", "agent_backend", "agent_model", "agent_step_limit", "agent_cost_limit", "agent_mcp",
    }),
)

for _kind in (LLM, TEMPLATE, PYTHON, AGENT):
    register_model_kind(_kind)
