"""
Static analysis of .prompt files — runs during graph construction.

Functions here are called before any model executes, to extract
dependencies, config, and promptdata variable names from raw template text.
"""

from __future__ import annotations

import json
import re

from jinja2 import Environment, Undefined


_REF_PATTERN = re.compile(r"""\bref\(\s*['"](\w+)['"]\s*\)""")
_PROMPTDATA_PATTERN = re.compile(r"""\bpromptdata\(\s*['"](\w+)['"]\s*\)""")


class _Empty:
    """Permissive stub returned during config-extraction dry-render.

    Supports attribute access, item access, iteration, and string conversion so
    that templates that immediately use the result (e.g. ``{{ ref('x').key }}``)
    don't raise during the dry-render.
    """
    def __str__(self):        return ""
    def __repr__(self):       return ""
    def __bool__(self):       return False
    def __len__(self):        return 0
    def __iter__(self):       return iter([])
    def __getattr__(self, _): return _Empty()
    def __getitem__(self, _): return _Empty()


def extract_dependencies(template_source: str) -> list[str]:
    """
    Return the list of model names referenced via ref() in *template_source*.

    Static scan — no Jinja rendering happens here.
    Duplicate references are deduplicated while preserving first-seen order.
    """
    seen: dict[str, None] = {}
    for match in _REF_PATTERN.finditer(template_source):
        seen[match.group(1)] = None
    return list(seen)


def detect_used_promptdata(template_source: str) -> list[str]:
    """
    Return a deduplicated list of variable names referenced via promptdata()
    in *template_source*, in first-seen order.

    Static scan — no Jinja rendering happens here.
    """
    seen: dict[str, None] = {}
    for match in _PROMPTDATA_PATTERN.finditer(template_source):
        seen[match.group(1)] = None
    return list(seen)


def extract_jinja_config(template_source: str) -> dict[str, str]:
    """
    Extract config set via an inline ``{{ config(...) }}`` call.

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
    Parse config for a .prompt file from an inline ``{{ config(...) }}`` call.

    Returns a dict of string keys to string values.
    """
    return extract_jinja_config(template_source)
