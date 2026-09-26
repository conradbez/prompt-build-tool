"""
LLM call resolution.

Looks for a user-provided client.py exposing ``llm_call(prompt: str) -> str``.
Raises a clear error if none is found — run `pbt init` to scaffold one.

client.py is also where a project registers its own model kinds, so pbt imports
it *before* parsing any .prompt file (see :func:`load_client_module`).  The
module is executed once per path and cached, so importing it early costs
nothing later.
"""

from __future__ import annotations

import importlib.util
import os
from types import ModuleType
from typing import Callable

CLIENT_FILENAME = "client.py"

_module_cache: dict[str, ModuleType] = {}


def client_path(models_dir: str) -> str:
    """Return the client.py path for *models_dir* (i.e. in its parent)."""
    return os.path.join(os.path.dirname(models_dir), CLIENT_FILENAME)


def load_client_module(models_dir: str) -> ModuleType | None:
    """Import the project's client.py, or return None when there isn't one.

    Executed once per path and cached.  Call this before loading models: any
    ``pbt.register_model_kind(...)`` in client.py must run before a .prompt file
    naming that kind is parsed, or the kind looks unknown.
    """
    candidate = client_path(models_dir)
    if not os.path.isfile(candidate):
        return None

    resolved = os.path.realpath(candidate)
    if resolved in _module_cache:
        return _module_cache[resolved]

    spec = importlib.util.spec_from_file_location("_pbt_user_client", candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _module_cache[resolved] = module
    return module


def try_load_client_module(models_dir: str) -> ModuleType | None:
    """Import client.py for its model-type registrations, ignoring any failure.

    Used at discovery time, where the client is wanted only for its side
    effects.  A client.py that cannot be imported is not reported here: the
    commands that actually need a backend call :func:`resolve_llm_call` later
    and surface the error there, with the message the user expects.
    """
    try:
        return load_client_module(models_dir)
    except Exception:
        return None


def resolve_llm_call(models_dir: str) -> Callable[[str], str]:
    """
    Search for client.py alongside models_dir (i.e. in its parent).
    If found and it defines ``llm_call``, return that function.
    Otherwise raise a FileNotFoundError with a helpful message.
    """
    module = load_client_module(models_dir)
    if module is not None:
        if hasattr(module, "llm_call"):
            return module.llm_call
        raise AttributeError(
            f"{client_path(models_dir)} was found but does not define an "
            "'llm_call(prompt: str) -> str' function."
        )

    raise FileNotFoundError(
        "No client.py found. Create one alongside your models/ directory with an "
        "'llm_call(prompt: str) -> str' function, or run `pbt init` to scaffold a starter project."
    )


def resolve_blob_store(models_dir: str):
    """Return the blob store client.py declares, or None.

    client.py opts in by defining ``blob_store`` — a
    :class:`~pbt.files.BlobStore` instance, or a zero-argument function
    returning one — to keep file outputs somewhere other than the run
    database (S3, a shared volume, …).
    """
    module = try_load_client_module(models_dir)
    provider = getattr(module, "blob_store", None) if module is not None else None
    if provider is None:
        return None
    import inspect

    from pbt.files import BlobStore

    is_factory = inspect.isfunction(provider) or inspect.isclass(provider)
    store = provider() if is_factory else provider
    if not isinstance(store, BlobStore):
        raise TypeError(
            f"{client_path(models_dir)}: blob_store must provide put(), get() and "
            f"exists(); got {type(store).__name__}."
        )
    return store


def resolve_classify_call(models_dir: str) -> Callable[[str, str], float] | None:
    """Return client.py's ``classify_call(state, question) -> float``, or None.

    Only classifier-judged tests need it; see :mod:`pbt.classifier`.
    """
    module = load_client_module(models_dir)
    return getattr(module, "classify_call", None) if module is not None else None


def resolve_test_judge(models_dir: str) -> str:
    """Return client.py's ``test_judge`` ("llm" or "classifier"), default "llm"."""
    from pbt.classifier import parse_judge

    module = load_client_module(models_dir)
    judge = getattr(module, "test_judge", None) if module is not None else None
    if judge is None:
        return "llm"
    return parse_judge(judge, f"{client_path(models_dir)} test_judge")
