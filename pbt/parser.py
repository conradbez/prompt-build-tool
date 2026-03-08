"""
Jinja2-based prompt parser with ref() support.

Special template function
-------------------------
{{ ref('model_name') }}
    Injects the LLM output of another model into this prompt.
    pbt uses calls to ref() to build the dependency graph before execution.

All standard Jinja2 features (loops, conditionals, filters, macros, etc.)
are available in addition to ref().
"""

import re
from typing import Callable

from jinja2 import Environment, StrictUndefined, Undefined


SKIP_SENTINEL = "SKIP THIS MODEL"
_SKIP_OUTPUT  = "SKIPPED THIS MODEL"

# Regex that finds every ref('...') or ref("...") call in raw template text.
# Used for static dependency extraction WITHOUT executing the template.
_REF_PATTERN = re.compile(r"""\bref\(\s*['"](\w+)['"]\s*\)""")


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
    extra_vars: dict | None = None,
    rag_call: "Callable[..., list[str]] | None" = None,
) -> str:
    """
    Render *template_source* as a Jinja2 template.

    Parameters
    ----------
    template_source:
        Raw contents of a .prompt file.
    model_outputs:
        Mapping of model_name → LLM output text for all upstream models.
    extra_vars:
        Additional variables injected into the template context.

    The ``ref(model_name)`` function is available inside templates and
    returns the corresponding entry from *model_outputs*.
    """
    env = _make_env()

    def ref(model_name: str) -> str:
        if model_name not in model_outputs:
            raise ValueError(
                f"ref('{model_name}') — model '{model_name}' has no output yet. "
                "This is likely a missing dependency or execution-order bug."
            )
        return model_outputs[model_name]

    def return_list_RAG_results(*args) -> list[str]:
        if rag_call is None:
            raise RuntimeError(
                "return_list_RAG_results() called but no rag_call was provided to render_prompt."
            )
        return rag_call(*args)

    def was_skipped(model_name: str) -> bool:
        return model_outputs.get(model_name) == _SKIP_OUTPUT

    context: dict = {
        "ref": ref,
        "return_list_RAG_results": return_list_RAG_results,
        "was_skipped": was_skipped,
        "skip_this_model": SKIP_SENTINEL,
    }
    if extra_vars:
        context.update(extra_vars)

    template = env.from_string(template_source)
    return template.render(**context)


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
