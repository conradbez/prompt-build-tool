"""
Model-kind registry — the single public place a new execution strategy is added.

A *model kind* is the value of ``{{ config(model_type="...") }}`` in a .prompt
file.  It selects how a rendered template becomes an output.  Everything around
that — rendering, the prompt cache, JSON parsing, validation, storage, skip
propagation — is owned by the executor and is identical for every kind (see
:mod:`promptmap.executor.executor`).

A kind is a frozen record of plain data and, at most, one function::

    import promptmap

    async def shout(rendered, call):
        response = await call.llm(rendered)
        return response.upper() + call.spec.config.get("suffix", "")

    promptmap.register_model_kind(promptmap.ModelKind("shout", shout, config_keys={"suffix"}))

or, equivalently, with the decorator::

    @promptmap.model_kind("shout", config_keys={"suffix"})
    async def shout(rendered, call):
        ...

There is no base class to subclass and no instance to construct: a kind holds
no per-model state, so it is data.  Fan-out is not a kind's concern: any
model sets ``config(each=...)`` and the executor runs its kind once per item.

Registering also declares the ``config()`` keys the kind consumes, so they stop
triggering :class:`~promptmap.executor.parser_initial.UnknownConfigKeyWarning` — there
is no second registry to keep in sync.

Where to register
-----------------
Anywhere that runs before the models are loaded.  ``client.py`` is the natural
home for a project-local kind: promptmap imports it before building the DAG precisely
so that kinds registered there are available.

Installed packages
------------------
A package that ships kinds declares them under the ``promptmap.model_kinds`` entry
point group, and ``pip install`` is then enough — nothing has to import it::

    [project.entry-points."promptmap.model_kinds"]
    shout = "my_pkg.shout"            # a module that registers on import
    other = "my_pkg:register"         # or a function promptmap calls once

promptmap loads that group lazily, on the first registry lookup, not at import time:
every plugin does ``import promptmap``, so loading plugins from inside promptmap's own
import would hand them a half-initialised module.  Set ``PROMPTMAP_DISABLE_PLUGINS=1``
to skip discovery (useful in tests).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from promptmap.secrets import Secrets

if TYPE_CHECKING:
    from typing import Iterable, Mapping

    from promptmap.model_spec import ModelSpec


@dataclass(frozen=True)
class ModelCall:
    """The preloaded means of producing one output, handed to an ``exec_fn``.

    Built by the executor per rendered prompt, with the model and its render
    state already bound — so an exec_fn takes the rendered text and nothing
    else it has to thread through.  Both callables here are cached, timed and
    skip-aware; a kind never implements any of that itself.
    """

    #: The model being run, for its name and its ``config()`` block.
    spec: "ModelSpec"

    #: Every upstream output produced so far, by model name.  A model that
    #: produced files appears as :class:`~promptmap.files.File` / ``Dir`` /
    #: ``Output`` objects (or a dict/list holding them); return the same types
    #: from an exec_fn to produce files.
    outputs: dict[str, Any]

    #: ``await call.llm(rendered)`` — send this prompt to the LLM backend.
    llm: Callable[[str], Awaitable[str]]

    #: ``await call.compute(rendered, compute=fn)`` — memoise arbitrary work
    #: against this prompt, for a kind that produces its output some other way.
    compute: Callable[..., Awaitable[Any]]

    #: Run-scoped credentials from ``run(secrets=...)``.  Read with
    #: ``call.secrets["NAME"]``.  Kept out of the repr so a printed call or a
    #: traceback never shows them.
    secrets: Secrets = field(default_factory=Secrets, repr=False)


#: ``async def exec_fn(rendered: str, call: ModelCall) -> Any``
ExecFn = Callable[[str, ModelCall], Any]


@dataclass(frozen=True)
class ModelKind:
    """How one ``model_type`` value turns a template into an output."""

    #: The ``config(model_type=...)`` value.  "" is the plain LLM call.
    name: str

    #: Produces the output from the rendered prompt.  ``None`` means the
    #: rendered text *is* the output — no LLM call, no execution.
    exec_fn: ExecFn | None = None

    #: config() keys this kind consumes, beyond the ones promptmap always knows.
    config_keys: frozenset[str] = field(default_factory=frozenset)

    #: Whether the run's global instruction may be rendered into this model's
    #: prompt.  False for kinds whose rendered template is not natural language
    #: (Python source, a pass-through value) and so would be corrupted by it.
    accepts_global_instruction: bool = True

    #: Secret names this kind reads from ``call.secrets``.  A run checks them
    #: before any model starts, so a missing key fails at once, not mid-run.
    required_secrets: frozenset[str] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: model_type value for a plain LLM call — what a .prompt file gets when it
#: sets no model_type at all.
DEFAULT_MODEL_TYPE = ""

_REGISTRY: dict[str, ModelKind] = {}

#: Entry point group installed packages register their kinds under.
ENTRY_POINT_GROUP = "promptmap.model_kinds"

_plugins_loaded = False


def load_plugins() -> None:
    """Import every installed package that registers promptmap model kinds.

    Runs once; later calls are no-ops.  A plugin that fails to load is logged
    and skipped so one broken package cannot take down every run.
    """
    global _plugins_loaded
    if _plugins_loaded or os.environ.get("PROMPTMAP_DISABLE_PLUGINS"):
        return
    # Set before loading: a plugin's own registry lookups must not recurse.
    _plugins_loaded = True

    from importlib.metadata import entry_points

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj = ep.load()  # importing runs the @promptmap.model_kind decorators
            if callable(obj):
                obj()  # optional explicit register() hook
        except Exception:
            logging.getLogger("promptmap").exception("promptmap plugin %r failed to load", ep.name)


def register_model_kind(kind: ModelKind) -> ModelKind:
    """Register *kind* under its own ``name``, replacing any kind of that name."""
    _REGISTRY[kind.name] = kind
    return kind


def model_kind(
    name: str,
    *,
    config_keys: "frozenset[str] | set[str] | None" = None,
    accepts_global_instruction: bool = True,
    secrets: "frozenset[str] | set[str] | None" = None,
):
    """Decorator form of :func:`register_model_kind`, wrapping an ``exec_fn``.

    ::

        @promptmap.model_kind("shout", config_keys={"suffix"})
        async def shout(rendered, call):
            return (await call.llm(rendered)).upper()

    *secrets* names the credentials the kind reads from ``call.secrets``; a
    run that selects the kind but lacks one fails before any model starts.

    The function is returned unchanged, so it stays directly callable and
    testable on its own.
    """
    def decorator(fn: ExecFn) -> ExecFn:
        register_model_kind(ModelKind(
            name=name,
            exec_fn=fn,
            config_keys=frozenset(config_keys or ()),
            accepts_global_instruction=accepts_global_instruction,
            required_secrets=frozenset(secrets or ()),
        ))
        return fn

    return decorator


def get_model_kind(name: str) -> ModelKind | None:
    """Return the kind registered under *name*, or None if there is none."""
    load_plugins()
    return _REGISTRY.get(name)


def known_model_kinds() -> set[str]:
    """Return every registered model_type value, excluding the unnamed default."""
    load_plugins()
    return {name for name in _REGISTRY if name != DEFAULT_MODEL_TYPE}


def missing_secrets(specs: "Iterable[ModelSpec]", secrets: "Mapping[str, str]") -> dict[str, set[str]]:
    """Map each secret the *specs*' kinds require but *secrets* lacks → kinds needing it."""
    missing: dict[str, set[str]] = {}
    for spec in specs:
        kind = get_model_kind(spec.model_type)
        for name in kind.required_secrets if kind is not None else ():
            if name not in secrets:
                missing.setdefault(name, set()).add(spec.model_type)
    return missing


def registered_config_keys() -> set[str]:
    """Return every config() key declared by a registered model kind."""
    load_plugins()
    keys: set[str] = set()
    for kind in _REGISTRY.values():
        keys.update(kind.config_keys)
    return keys
