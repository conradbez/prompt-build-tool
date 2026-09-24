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
``agent``            hand the rendered text to mini-swe-agent as its task

Importing this module registers them; :mod:`pbt` does that on import.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
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
    """Run mini-swe-agent with the rendered template as its task.

    The agent works in ``agent_dir`` (created if missing), running shell
    commands there until it submits.  The output is a dict::

        {"output": <what the agent submitted>,
         "logs": <the full message trajectory>,
         "time_run": <seconds the agent ran>}

    so a downstream model reads ``ref('fix')['output']``.

    Config keys:

    ``agent_dir``         working directory (required)
    ``agent_model``       litellm model name; defaults to ``MSWEA_MODEL_NAME``
    ``agent_step_limit``  max LLM calls, 0 for none (default 0)
    ``agent_cost_limit``  max spend in dollars, 0 for none (default 3)

    Needs ``pip install mini-swe-agent``.  Results are cached on the rendered
    prompt like any other kind, so an unchanged task does not re-run.
    """
    raw = await call.compute(
        rendered, compute=lambda: asyncio.to_thread(_exec_agent, rendered, call)
    )
    return json.loads(raw)


def _exec_agent(task: str, call: ModelCall) -> str:
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
    workdir = config.get("agent_dir")
    if not workdir:
        raise ValueError(
            f"Model '{call.spec.name}': model_type=\"agent\" needs "
            "config(agent_dir=\"...\") — the directory the agent works in."
        )
    workdir = Path(workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

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

    started = time.monotonic()
    result = agent.run(task)
    time_run = round(time.monotonic() - started, 3)

    return json.dumps(
        {"output": result.get("submission", ""), "logs": agent.messages, "time_run": time_run},
        default=str,
    )


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

#: mini-swe-agent working in a directory, with the rendered text as its task.
AGENT = ModelKind(
    name="agent",
    exec_fn=run_agent,
    config_keys=frozenset({"agent_dir", "agent_model", "agent_step_limit", "agent_cost_limit"}),
)

for _kind in (LLM, TEMPLATE, PYTHON, AGENT):
    register_model_kind(_kind)
