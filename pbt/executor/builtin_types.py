"""
The model types pbt ships with.

Each is a small strategy: render something, produce a value.  Everything around
that — the prompt cache, JSON parsing, validation, storage, skip propagation —
belongs to the executor and is identical for every type, including the ones you
register yourself.

Importing this module registers them; :mod:`pbt.model_types` does that on first
use of the registry.
"""

from __future__ import annotations

import asyncio
import io
import json
from contextlib import redirect_stdout
from typing import Any

from pbt.executor.run_context import RunContext, parse_json_output
from pbt.model_spec import ModelSpec
from pbt.model_types import BaseModelType, register_model_type


class LLMModelType(BaseModelType):
    """The default: render the template, send it to the LLM."""

    async def execute(self, spec: ModelSpec, ctx: RunContext) -> Any:
        rendered, state = ctx.render(spec)
        return await ctx.call_llm(rendered, spec, state)


class TemplateModelType(BaseModelType):
    """Render the template and use it as the output — no LLM call.

    For nodes that only reshape what upstream models already produced: a
    header, a merge of two outputs, a pass-through.  A global instruction is
    never applied, since the rendered text is the output itself rather than a
    prompt for a model to answer.
    """

    accepts_global_instruction = False

    async def execute(self, spec: ModelSpec, ctx: RunContext) -> Any:
        rendered, _ = ctx.render(spec)
        return rendered


class LoopModelType(BaseModelType):
    """Fan out over a JSON list from an upstream model, one LLM call per item.

    ``ref('<the list model>')`` yields the current item inside the template, so
    a loop model needs no syntax of its own.  Calls run concurrently and the
    outputs are collected back into a list, in input order.
    """

    config_keys = frozenset({"loop_over"})

    async def execute(self, spec: ModelSpec, ctx: RunContext) -> Any:
        dep_name, items = self._resolve_list_dep(spec, ctx)
        ctx.note(spec, f"[loop over {len(items)} items from '{dep_name}']")

        renders = [
            ctx.render(spec, extra_outputs={dep_name: item}, primary=False)
            for item in items
        ]

        async def run_one(rendered: str, state) -> Any:
            output = await ctx.call_llm(rendered, spec, state)
            if state.skip_value is None and spec.output_format == "json":
                return parse_json_output(output)
            return output

        return list(await asyncio.gather(
            *(run_one(rendered, state) for rendered, state in renders)
        ))

    @staticmethod
    def _resolve_list_dep(spec: ModelSpec, ctx: RunContext) -> tuple[str, list]:
        """Pick the upstream dependency to fan out over."""
        list_deps = {
            dep: ctx.outputs[dep]
            for dep in spec.depends_on
            if isinstance(ctx.outputs.get(dep), list)
        }

        pinned = spec.config.get("loop_over", "")
        if pinned:
            if pinned not in spec.depends_on:
                raise ValueError(
                    f"Loop model '{spec.name}': loop_over='{pinned}' is not a "
                    f"dependency of this model. Dependencies: {spec.depends_on!r}."
                )
            if pinned not in list_deps:
                raise ValueError(
                    f"Loop model '{spec.name}': loop_over='{pinned}' does not return "
                    "a JSON list. Ensure it has output_format='json' and returns a list."
                )
            return pinned, list_deps[pinned]

        if not list_deps:
            raise ValueError(
                f"Loop model '{spec.name}': no upstream dependency returns a JSON list. "
                "Ensure an upstream model has output_format='json' and returns a list."
            )
        if len(list_deps) > 1:
            raise ValueError(
                f"Loop model '{spec.name}': multiple dependencies return lists: "
                f"{list(list_deps)}. Add loop_over='model_name' to config() to disambiguate."
            )
        return next(iter(list_deps.items()))


class PythonModelType(BaseModelType):
    """Render the template as Python source and run it with ``exec()``.

    Output is whatever the code prints; failing that, a variable named
    ``output``.  Upstream outputs are available as ``ref('name')`` and in the
    ``model_outputs`` dict.  The code runs in-process with full builtins — a
    .prompt file using this type is trusted code.
    """

    accepts_global_instruction = False

    async def execute(self, spec: ModelSpec, ctx: RunContext) -> Any:
        rendered, state = ctx.render(spec)
        # Goes through the same cache as an LLM call, so unchanged code does not
        # re-execute on the next run.
        return await ctx.cached(
            rendered, spec, state, compute=lambda: self._exec(rendered, spec, ctx)
        )

    @staticmethod
    def _exec(rendered: str, spec: ModelSpec, ctx: RunContext) -> str:
        namespace: dict = {
            "model_outputs": ctx.outputs,
            "ref": lambda name: ctx.outputs.get(name),
        }
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exec(compile(rendered, f"<{spec.name}>", "exec"), namespace)  # noqa: S102

        printed = stdout.getvalue()
        if printed:
            return printed.rstrip("\n")
        if "output" in namespace:
            value = namespace["output"]
            return json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        return ""


class QualityCheckModelType(BaseModelType):
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

    config_keys = frozenset({"quality_retries", "quality_pass_marker"})

    def expand(
        self, spec: ModelSpec, all_specs: dict[str, ModelSpec]
    ) -> list[ModelSpec]:
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

            # The retry re-runs the checked model, but skips (passing the
            # previous output straight through) when the check said PASS, and
            # otherwise offers the check's critique as feedback.
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


register_model_type("", LLMModelType)
register_model_type("template", TemplateModelType)
register_model_type("loop", LoopModelType)
register_model_type("execute_python", PythonModelType)
register_model_type("quality_check", QualityCheckModelType)
