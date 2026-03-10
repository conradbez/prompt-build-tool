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
are available in addition to ref() and promptdata().
"""

import re
from pathlib import Path
from typing import Callable

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Undefined


SKIP_SENTINEL = "SKIP THIS MODEL"
_SKIP_OUTPUT  = "SKIPPED THIS MODEL"

# Regex that finds every ref('...') or ref("...") call in raw template text.
# Used for static dependency extraction WITHOUT executing the template.
_REF_PATTERN = re.compile(r"""\bref\(\s*['"](\w+)['"]\s*\)""")

# Regex that finds every promptdata('...') or promptdata("...") call.
# Used for static detection WITHOUT executing the template.
_PROMPTDATA_PATTERN = re.compile(r"""\bpromptdata\(\s*['"](\w+)['"]\s*\)""")

# Regex to extract pbt config block: {# pbt:config ... #} at the top of file.
_CONFIG_PATTERN = re.compile(
    r"^\s*\{#\s*pbt:config\s*(.*?)\s*#\}", re.DOTALL
)


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


def _parse_static_config(template_source: str) -> dict[str, str]:
    """Parse the ``{# pbt:config ... #}`` comment block, returning str→str pairs."""
    match = _CONFIG_PATTERN.search(template_source)
    if not match:
        return {}
    config: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            config[key.strip()] = value.strip()
    return config


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
            captured[k] = str(v) if not isinstance(v, list) else ",".join(str(i) for i in v)
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
        "skip_this_model": "",
    }

    try:
        env.from_string(template_source).render(**context)
    except Exception:
        pass  # best-effort; partial capture is fine

    return captured


def parse_model_config(template_source: str) -> dict:
    """
    Parse config for a .prompt file from either source:

    1. A ``{# pbt:config ... #}`` comment block (YAML-like, legacy)::

           {# pbt:config
           output_format: json
           #}

    2. An inline ``{{ config(...) }}`` Jinja call (dbt-style)::

           {{ config(output_format="json", tags="article") }}

    Both are merged; the inline ``config()`` call takes precedence on conflicts.
    Returns a dict of string keys to string values.
    """
    merged = _parse_static_config(template_source)
    merged.update(extract_jinja_config(template_source))
    return merged


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
    templates_dir: "Path | None" = None,
) -> str:
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
    templates_dir:
        Optional path to the global templates directory. When provided,
        templates can include shared snippets via ``{% include 'name.j2' %}``.

    The ``ref(model_name)`` function is available inside templates and
    returns the corresponding entry from *model_outputs*.
    The ``promptdata(name)`` function returns runtime variables.
    """
    env = _make_env(templates_dir)
    _promptdata = promptdata or {}

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
        return model_outputs.get(model_name) == _SKIP_OUTPUT

    context: dict = {
        "ref": ref,
        "promptdata": _promptdata_fn,
        "return_list_RAG_results": return_list_RAG_results,
        "was_skipped": was_skipped,
        "skip_this_model": SKIP_SENTINEL,
        "config": lambda **_: "",   # no-op during real render; config already parsed

    }

    template = env.from_string(template_source)
    return template.render(**context)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_env(templates_dir: "Path | None" = None) -> Environment:
    """Return a Jinja2 Environment configured for prompt files.

    If *templates_dir* is provided and exists, a ``FileSystemLoader`` is
    configured so templates can include shared snippets::

        {% include 'system_prompt.j2' %}
    """
    loader = (
        FileSystemLoader(str(templates_dir))
        if templates_dir is not None and templates_dir.is_dir()
        else None
    )
    return Environment(
        loader=loader,
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
