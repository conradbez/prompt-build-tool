"""
LLM call resolution.

Looks for a user-provided client.py exposing ``llm_call(prompt: str) -> str``.
Falls back to the built-in Gemini implementation if none is found.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Callable

_DEFAULT_MODEL = "gemini-3-flash-preview"


def resolve_llm_call(models_dir: str) -> Callable[[str], str]:
    """
    Search for client.py alongside models_dir (i.e. in its parent), then
    inside models_dir itself for backwards compatibility.
    If found and it defines ``llm_call``, return that function.
    Otherwise return the built-in Gemini implementation.
    """
    for candidate in [
        os.path.join(os.path.dirname(models_dir), "client.py"),
        os.path.join(models_dir, "client.py"),
    ]:
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

    return _gemini_llm_call


def _gemini_llm_call(prompt: str, files: list[str] | None = None) -> str:
    try:
        from google import genai
        from pathlib import Path
    except ImportError as exc:
        raise ImportError(
            "google-genai is required to run prompts. "
            "Install it with: pip install google-genai"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Obtain a key at https://aistudio.google.com/app/apikey "
            "and export it:\n\n  export GEMINI_API_KEY=your_key_here"
        )
    model_name = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)
    client = genai.Client(api_key=api_key)

    if files:
        contents: list = [prompt]
        for file_path in files:
            p = Path(file_path)
            file_data = client.files.upload(path=p)
            contents.append(file_data)
        return client.models.generate_content(model=model_name, contents=contents).text

    return client.models.generate_content(model=model_name, contents=prompt).text
