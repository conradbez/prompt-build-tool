"""
Jinja2 template rendering — runs during model execution.

``render_prompt`` is called once per model immediately before the LLM call,
injecting upstream outputs via ref() and evaluating skip logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from jinja2 import Environment, StrictUndefined, meta


class _Meta:
    """Open namespace for template-settable model metadata.

    Attributes are set dynamically (e.g. ``feedback_from_previous_run`` in
    quality retry nodes).  Any unset attribute returns ``None`` so
    ``{% if model.meta.feedback_from_previous_run %}`` is always safe.
    """
    def __getattr__(self, name: str):
        return None

    def _set(self, **kwargs) -> str:
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)
        return ""


@dataclass
class _ModelContext:
    """Metadata about the current model, available as ``{{ model }}`` in templates."""
    name: str = ""
    meta: _Meta = field(default_factory=_Meta)


@dataclass
class _RenderState:
    """Mutable state set by skip functions during template rendering.

    ``skip_value is None``  → model runs normally.
    ``skip_value is not None``  → skip the LLM call; use the value as output.
    ``skip_downstream``  → also propagate a skip signal to all downstream models.
    """
    skip_value: str | None = None
    skip_downstream: bool = False


def render_prompt(
    template_source: str,
    model_outputs: dict[str, str],
    promptdata: dict | None = None,
    rag_call: "Callable[..., list[str]] | None" = None,
    prompt_skipped_models: "set[str] | None" = None,
    model_name: str = "",
    global_instruction: str | None = None,
) -> "tuple[str, _RenderState]":
    """
    Render *template_source* as a Jinja2 template.

    Parameters
    ----------
    template_source:
        Raw contents of a .prompt file.
    model_outputs:
        Mapping of model_name → LLM output text for all upstream models.
    promptdata:
        Optional dict of runtime variables, injected via promptdata("name").
        Returns None for missing keys so {% if promptdata('x') %} is safe.
    global_instruction:
        Optional prompt text rendered into every model prompt (see
        ``pbt.global_instruction``).  It is rendered with this model's own
        Jinja context, so ``promptdata()``, ``model`` and ``was_skipped()``
        work inside it.  Reference ``{{ prompt }}`` in it to place the model
        body explicitly; otherwise the instruction is prepended.  Ignored when
        a skip function fired, since no LLM call will be made.

    Returns
    -------
    A ``(rendered_prompt, render_state)`` tuple.  ``render_state.skip_value``
    is ``None`` when the model should run normally, or a string to use as the
    model output directly without calling the LLM.
    """
    env = _make_env()
    _promptdata = promptdata or {}
    state = _RenderState()
    model_context = _ModelContext(name=model_name)

    def ref(model_name: str) -> str:
        if model_name not in model_outputs:
            raise ValueError(
                f"ref('{model_name}') — model '{model_name}' has no output yet. "
                "This is likely a missing dependency or execution-order bug."
            )
        return model_outputs[model_name]

    def _promptdata_fn(name: str):
        return _promptdata.get(name)

    def return_list_RAG_results(*args) -> list[str]:
        if rag_call is None:
            raise RuntimeError(
                "return_list_RAG_results() called but no rag_call was provided to render_prompt."
            )
        return rag_call(*args)

    def was_skipped(model_name: str) -> bool:
        return model_name in (prompt_skipped_models or set())

    context: dict = {
        "ref": ref,
        "promptdata": _promptdata_fn,
        "return_list_RAG_results": return_list_RAG_results,
        "was_skipped": was_skipped,
        "config": lambda **_: "",   # no-op during real render; config already parsed
        "model": model_context,
    }

    def skip_and_set_to_value(value) -> str:
        """Skip the LLM call and use the Jinja-rendered *value* as both prompt and output."""
        rendered_value = env.from_string(str(value)).render(**{**context, "skip_and_set_to_value": lambda nested_value="": str(nested_value)})
        state.skip_value = rendered_value
        return rendered_value

    def skip_this_and_downstream(value="") -> str:
        """Skip this model's LLM call and signal all downstream models to skip too."""
        rendered_value = env.from_string(str(value)).render(**{**context, "skip_this_and_downstream": lambda nested_value="": str(nested_value)})
        state.skip_value = rendered_value
        state.skip_downstream = True
        return rendered_value

    context["skip_and_set_to_value"] = skip_and_set_to_value
    context["skip_this_and_downstream"] = skip_this_and_downstream

    template = env.from_string(template_source)
    rendered = template.render(**context)

    if global_instruction and state.skip_value is None:
        rendered = _apply_global_instruction(env, global_instruction, rendered, context)

    return rendered, state


def _apply_global_instruction(
    env: Environment, source: str, body: str, context: dict
) -> str:
    """Render *source* with the model's context and combine it with *body*.

    ``{{ prompt }}`` inside the instruction is the rendered model body; when the
    instruction references it the result is used as-is (wrapper semantics),
    otherwise the instruction is prepended to the body.
    """
    def _ref_unavailable(name: str = "") -> str:
        raise ValueError(
            f"ref('{name}') is not available in a global instruction. The global "
            f"instruction is rendered into every model, so this would make every "
            f"model depend on '{name}' — including '{name}' itself. Use "
            f"promptdata() for values that vary between runs."
        )

    global_context = {**context, "ref": _ref_unavailable, "prompt": body}
    rendered_global = env.from_string(source).render(**global_context)

    if not rendered_global.strip():
        return body
    if "prompt" in meta.find_undeclared_variables(env.parse(source)):
        return rendered_global
    return rendered_global.rstrip("\n") + "\n\n" + body


def _make_env() -> Environment:
    """Return a Jinja2 Environment configured for prompt files."""
    return Environment(
        keep_trailing_newline=True,
        undefined=StrictUndefined,
        block_start_string="{%",
        block_end_string="%}",
        variable_start_string="{{",
        variable_end_string="}}",
        comment_start_string="{#",
        comment_end_string="#}",
        trim_blocks=True,
        lstrip_blocks=True,
    )
