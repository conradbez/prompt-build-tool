"""
LLM call resolution.

Looks for a user-provided client.py exposing ``llm_call(prompt: str) -> str``.
Raises a clear error if none is found — run `pbt init` to scaffold one.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Callable


def resolve_llm_call(models_dir: str) -> Callable[[str], str]:
    """
    Search for client.py alongside models_dir (i.e. in its parent).
    If found and it defines ``llm_call``, return that function.
    Otherwise raise a FileNotFoundError with a helpful message.
    """
    for candidate in [os.path.join(os.path.dirname(models_dir), "client.py")]:
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("_pbt_user_client", candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "llm_call"):
                return mod.llm_call
            raise AttributeError(
                f"{candidate} was found but does not define an "
                "'llm_call(prompt: str) -> str' function."
            )

    raise FileNotFoundError(
        "No client.py found. Create one alongside your models/ directory with an "
        "'llm_call(prompt: str) -> str' function, or run `pbt init` to scaffold a starter project."
    )
