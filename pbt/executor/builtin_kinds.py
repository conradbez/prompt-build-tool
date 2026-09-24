"""
The model kinds pbt ships with.

Each is a :class:`~pbt.model_types.ModelKind` record: some data, and at most one
function that turns a rendered prompt into a value.  Everything around that —
rendering, the prompt cache, JSON parsing, validation, storage, skip
propagation — belongs to the executor and is identical for every kind,
including the ones you register yourself.

Three of the five need no execution code at all:

``""`` (plain LLM)   send the rendered prompt to the backend
``template``         the rendered text *is* the output (``exec_fn=None``)
``loop``             the plain LLM call, once per item (``fan_out=True``)
``execute_python``   run the rendered text as Python
``quality_check``    pure DAG-build-time rewrite (``expand_fn``)

Importing this module registers them; :mod:`pbt` does that on import.
"""

from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from pbt.files import Dir, File, FileOutputError, Output, contains_files

from pbt.model_spec import ModelSpec
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


# ---------------------------------------------------------------------------
# expand_fn
# ---------------------------------------------------------------------------

def expand_quality_check(
    spec: ModelSpec, all_specs: dict[str, ModelSpec]
) -> list[ModelSpec]:
    """Expand one node into a check-and-retry chain at DAG-build time.

    Given ``article → article_quality → summary`` with ``quality_retries="2"``,
    the graph becomes::

        article_quality_1  ← quality check on the original article
        article_1          ← retry (skips if check 1 passed)
        article_quality_2  ← quality check on article_1
        article_2          ← retry (skips if check 2 passed)
        article_quality    ← terminal pass-through, keeps the original name

    The node's own template is the check prompt and must ``ref()`` exactly one
    upstream model — the one being checked.  Because the terminal node keeps the
    declared name, nothing downstream knows the expansion happened.
    """
    retries = spec.config_int("quality_retries", 2)
    pass_marker = spec.config.get("quality_pass_marker", "PASS")

    target_deps = [dep for dep in spec.depends_on if dep in all_specs]
    if len(target_deps) != 1:
        raise ValueError(
            f"Quality check node '{spec.name}' must depend on exactly one "
            f"upstream model; got depends_on={spec.depends_on!r}."
        )
    target = all_specs[target_deps[0]]

    check_config = {k: v for k, v in spec.config.items() if k != "model_type"}
    retry_config = {k: v for k, v in target.config.items() if k != "model_type"}

    nodes: list[ModelSpec] = []
    previous = target.name  # most recent version of the checked model

    for attempt in range(1, retries + 1):
        check_name = f"{spec.name}_{attempt}"
        retry_name = f"{target.name}_{attempt}"

        nodes.append(spec.derive(
            name=check_name,
            model_type="",
            config=check_config,
            source=(
                spec.source if attempt == 1
                else _retarget_ref(spec.source, target.name, previous)
            ),
            depends_on=list(spec.depends_on) if attempt == 1 else [previous],
        ))

        # The retry re-runs the checked model, but skips (passing the previous
        # output straight through) when the check said PASS, and otherwise
        # offers the check's critique as feedback.
        skip_block = (
            f"{{% if '{pass_marker}' in ref('{check_name}')|upper %}}"
            f"{{{{ skip_and_set_to_value(ref('{previous}')) }}}}"
            f"{{% endif %}}\n"
        )
        feedback_block = (
            f"{{% set _ = model.meta._set("
            f"feedback_from_previous_run=ref('{check_name}')) %}}\n"
        )
        nodes.append(target.derive(
            name=retry_name,
            model_type="",
            config=retry_config,
            source=skip_block + feedback_block + target.source,
            depends_on=list(dict.fromkeys(
                list(target.depends_on)
                + [check_name]
                + ([previous] if previous != target.name else [])
            )),
        ))
        previous = retry_name

    # Terminal node, keeping the declared name so downstream ref() still works.
    nodes.append(spec.derive(
        model_type="template",
        config={},
        source=f"{{{{ ref('{previous}') }}}}",
        depends_on=[previous],
        promptdata_used=[],
        promptfiles_used=[],
    ))
    return nodes


def _retarget_ref(source: str, old_name: str, new_name: str) -> str:
    """Point a check template's ref() at the latest retry instead of the original."""
    return (
        source
        .replace(f"ref('{old_name}')", f"ref('{new_name}')")
        .replace(f'ref("{old_name}")', f'ref("{new_name}")')
    )


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

#: Fan out over a JSON list from an upstream model, one LLM call per item.
#: ``ref('<the list model>')`` yields the current item inside the template, so a
#: loop model needs no syntax of its own.
LOOP = ModelKind(
    name="loop",
    exec_fn=call_the_llm,
    fan_out=True,
    config_keys=frozenset({"loop_over"}),
)

PYTHON = ModelKind(
    name="execute_python",
    exec_fn=run_python,
    accepts_global_instruction=False,
)

QUALITY_CHECK = ModelKind(
    name="quality_check",
    exec_fn=call_the_llm,
    expand_fn=expand_quality_check,
    config_keys=frozenset({"quality_retries", "quality_pass_marker"}),
)

for _kind in (LLM, TEMPLATE, LOOP, PYTHON, QUALITY_CHECK):
    register_model_kind(_kind)
