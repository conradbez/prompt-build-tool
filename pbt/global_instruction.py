"""
Global instruction resolution.

A *global instruction* is a snippet of prompt text rendered into every model
prompt in the run — pbt's analogue of dbt's ``query-comment``, which dbt renders
through Jinja and injects into every query it sends to the warehouse.

Two sources, in precedence order:

1. The ``global_instruction=`` argument to :func:`pbt.run` — a string, or a
   zero-argument callable returning a string (build it from your own config,
   environment, or registry at import time).
2. A ``global.prompt`` file alongside the models directory, discovered the same
   way ``client.py`` and ``rag.py`` are.

Models opt out individually with ``{{ config(global_instruction=False) }}``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

GLOBAL_INSTRUCTION_FILENAME = "global.prompt"


def resolve_global_instruction(models_dir: str) -> str | None:
    """
    Search for global.prompt alongside *models_dir* (i.e. in its parent).

    Returns its source text, or None when the file does not exist — an absent
    global.prompt is the normal case, not an error.
    """
    candidate = os.path.join(os.path.dirname(models_dir), GLOBAL_INSTRUCTION_FILENAME)
    if os.path.isfile(candidate):
        return Path(candidate).read_text(encoding="utf-8")
    return None


def coerce_global_instruction(value: "str | Callable[[], str] | None") -> str | None:
    """
    Normalise a ``global_instruction=`` argument to source text or None.

    Callables are invoked once, here, so the resulting text is fixed for the
    whole run rather than re-evaluated per model.
    """
    if value is None:
        return None
    if callable(value):
        value = value()
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            "global_instruction must be a string, a callable returning a string, "
            f"or None — got {type(value).__name__}."
        )
    return value or None
