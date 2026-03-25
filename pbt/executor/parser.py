"""
Jinja2-based prompt parser with ref() and promptdata() support.

Special template functions
--------------------------
{{ ref('model_name') }}
    Injects the LLM output of another model into this prompt.
    pbt uses calls to ref() to build the dependency graph before execution.

{{ promptdata('var_name') }}
    Injects a runtime variable passed via --promptdata or the Python API.
    Returns None if the variable was not provided (so {% if promptdata('x') %}
    works safely).

All standard Jinja2 features (loops, conditionals, filters, macros, etc.)
are available in addition to ref(), promptdata(), and skip helpers.
"""

import json
import re
from dataclasses import dataclass
from typing import Callable

from jinja2 import Environment, StrictUndefined, Undefined



@dataclass
class _RenderState:
    """Mutable state set by skip functions during template rendering.

    ``skip_value is None``  → model runs normally.
    ``skip_value is not None``  → skip the LLM call; use the value as output.
    ``skip_downstream``  → also propagate a skip signal to all downstream models.
    """
    skip_value: str | None = None
    skip_downstream: bool = False

# Regex that finds every ref('...') or ref("...") call in raw template text.
# Used for static dependency extraction WITHOUT executing the template.
_REF_PATTERN = re.compile(r"""\bref\(\s*['"](\w+)['"]\s*\)""")

# Regex that finds every promptdata('...') or promptdata("...") call.
# Used for static detection WITHOUT executing the template.
_PROMPTDATA_PATTERN = re.compile(r"""\bpromptdata\(\s*['"](\w+)['"]\s*\)""")

class _Empty:
    """Permissive stub returned by ref() and other functions during config extraction.

    Supports attribute access, item access, iteration, and string conversion so
    that templates that immediately use the result (e.g. ``{{ ref('x').key }}``)
    don't raise during the config-extraction dry-render.
    """
    def __str__(self):       return ""
    def __repr__(self):      return ""
    def __bool__(self):      return False
    def __len__(self):       return 0
    def __iter__(self):      return iter([])
    def __getattr__(self, _): return _Empty()
    def __getitem__(self, _): return _Empty()


def detect_used_promptdata(template_source: str) -> list[str]:
    """
    Return a deduplicated list of variable names referenced via promptdata()
    in *template_source*, in first-seen order.

    This is a static scan — no Jinja rendering happens here.
    """
    seen: dict[str, None] = {}
    for match in _PROMPTDATA_PATTERN.finditer(template_source):
        seen[match.group(1)] = None
    return list(seen)
def extract_jinja_config(template_source: str) -> dict[str, str]:
    """
    Extract config set via an inline ``{{ config(...) }}`` call in the template.

    The ``config()`` function works like dbt's — call it anywhere in the file::

        {{ config(output_format="json", tags="article") }}

    Values are coerced to strings to match the existing config dict convention.
    Returns an empty dict if no ``config()`` call is present or if rendering fails.
    """
    captured: dict[str, str] = {}

    def _config(**kwargs) -> str:
        for k, v in kwargs.items():
            captured[k] = json.dumps(v) if isinstance(v, list) else str(v)
        return ""

    env = Environment(
        keep_trailing_newline=True,
        undefined=Undefined,   # lenient — missing vars become empty strings
        block_start_string="{%",
        block_end_string="%}",
        variable_start_string="{{",
        variable_end_string="}}",
        comment_start_string="{#",
        comment_end_string="#}",
        trim_blocks=True,
        lstrip_blocks=True,
    )

    context: dict = {
        "config": _config,
        "ref": lambda *a, **kw: _Empty(),
        "promptdata": lambda *a, **kw: None,
        "return_list_RAG_results": lambda *a, **kw: [],
        "was_skipped": lambda *a, **kw: False,
        "skip_and_set_to_value": lambda value="": "",
        "skip_this_and_downstream": lambda value="": "",
    }

    try:
        env.from_string(template_source).render(**context)
    except Exception:
        pass  # best-effort; partial capture is fine

    return captured


def parse_model_config(template_source: str) -> dict:
    """
    Parse config for a .prompt file from an inline ``{{ config(...) }}`` call::

           {{ config(output_format="json", tags="article") }}

    Returns a dict of string keys to string values.
    """
    return extract_jinja_config(template_source)


def extract_dependencies(template_source: str) -> list[str]:
    """
    Return the list of model names referenced via ref() in *template_source*.

    This is a static scan — no Jinja rendering happens here.
    Duplicate references are deduplicated while preserving first-seen order.
    """
    seen: dict[str, None] = {}
    for match in _REF_PATTERN.finditer(template_source):
        seen[match.group(1)] = None
    return list(seen)


def render_prompt(
    template_source: str,
    model_outputs: dict[str, str],
    promptdata: dict | None = None,
    rag_call: "Callable[..., list[str]] | None" = None,
    prompt_skipped_models: "set[str] | None" = None,
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

    Returns
    -------
    A ``(rendered_prompt, render_state)`` tuple.  ``render_state.skip_value``
    is ``None`` when the model should run normally, or a string to use as the
    model output directly without calling the LLM.

    The ``ref(model_name)`` function is available inside templates and
    returns the corresponding entry from *model_outputs*.
    The ``promptdata(name)`` function returns runtime variables.
    """
    env = _make_env()
    _promptdata = promptdata or {}
    state = _RenderState()

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
    return template.render(**context), state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_env() -> Environment:
    """Return a Jinja2 Environment configured for prompt files."""
    return Environment(
        # Keep newlines in templates so multi-paragraph prompts render cleanly
        keep_trailing_newline=True,
        # Raise on unknown variables so typos are caught early
        undefined=StrictUndefined,
        # Standard Jinja2 delimiters — easy to read in plain-text prompts
        block_start_string="{%",
        block_end_string="%}",
        variable_start_string="{{",
        variable_end_string="}}",
        comment_start_string="{#",
        comment_end_string="#}",
        # Strip leading whitespace on blocks to keep prompts tidy
        trim_blocks=True,
        lstrip_blocks=True,
    )
