"""
RAG call resolution.

Looks for a user-provided rag.py alongside models_dir exposing
``do_RAG(*args) -> list[str] | str``.

If found, returns that function wrapped so the result is always a list[str].
If not found, raises an informative error when the function is actually called.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Callable


def resolve_rag_call(models_dir: str) -> Callable[..., list[str]]:
    """
    Search for rag.py alongside models_dir (i.e. in its parent), then
    inside models_dir itself for backwards compatibility.
    If found and it defines ``do_RAG``, return a wrapper that always
    returns list[str].  Otherwise return a stub that raises on call.
    """
    for candidate in [
        os.path.join(os.path.dirname(models_dir), "rag.py"),
        os.path.join(models_dir, "rag.py"),
    ]:
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("_pbt_user_rag", candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not hasattr(mod, "do_RAG"):
                raise AttributeError(
                    f"{candidate} was found but does not define a "
                    "'do_RAG(*args) -> list[str] | str' function."
                )
            raw_fn = mod.do_RAG

            def _wrapped(*args, _fn=raw_fn) -> list:
                result = _fn(*args)
                if result is False or result is None:
                    return [False]
                if isinstance(result, str):
                    return [result]
                return list(result)

            return _wrapped

    def _missing(*args) -> list[str]:
        raise RuntimeError(
            "return_list_RAG_results() was called but no rag.py was found.\n"
            "Create rag.py (alongside your models/ directory) with a 'do_RAG(*args) -> list[str] | str' function."
        )

    return _missing
