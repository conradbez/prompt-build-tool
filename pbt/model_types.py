"""
Model-kind registry — the single public place a new execution strategy is added.

A *model kind* is the value of ``{{ config(model_type="...") }}`` in a .prompt
file.  It selects how a rendered template becomes an output.  Everything around
that — rendering, the prompt cache, JSON parsing, validation, storage, skip
propagation — is owned by the executor and is identical for every kind (see
:mod:`pbt.executor.executor`).

A kind is a frozen record of plain data and, at most, one function::

    import pbt

    async def shout(rendered, call):
        response = await call.llm(rendered)
        return response.upper() + call.spec.config.get("suffix", "")

    pbt.register_model_kind(pbt.ModelKind("shout", shout, config_keys={"suffix"}))

or, equivalently, with the decorator::

    @pbt.model_kind("shout", config_keys={"suffix"})
    async def shout(rendered, call):
        ...

There is no base class to subclass and no instance to construct: a kind holds
no per-model state, so it is data.

Registering also declares the ``config()`` keys the kind consumes, so they stop
triggering :class:`~pbt.executor.parser_initial.UnknownConfigKeyWarning` — there
is no second registry to keep in sync.

Where to register
-----------------
Anywhere that runs before the models are loaded.  ``client.py`` is the natural
home for a project-local kind: pbt imports it before building the DAG precisely
so that kinds registered there are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from pbt.model_spec import ModelSpec


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
    #: produced files appears as :class:`~pbt.files.File` / ``Dir`` /
    #: ``Output`` objects (or a dict/list holding them); return the same types
    #: from an exec_fn to produce files.
    outputs: dict[str, Any]

    #: ``await call.llm(rendered)`` — send this prompt to the LLM backend.
    llm: Callable[[str], Awaitable[str]]

    #: ``await call.compute(rendered, compute=fn)`` — memoise arbitrary work
    #: against this prompt, for a kind that produces its output some other way.
    compute: Callable[..., Awaitable[Any]]


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

    #: config() keys this kind consumes, beyond the ones pbt always knows.
    config_keys: frozenset[str] = field(default_factory=frozenset)

    #: Whether the run's global instruction may be rendered into this model's
    #: prompt.  False for kinds whose rendered template is not natural language
    #: (Python source, a pass-through value) and so would be corrupted by it.
    accepts_global_instruction: bool = True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: model_type value for a plain LLM call — what a .prompt file gets when it
#: sets no model_type at all.
DEFAULT_MODEL_TYPE = ""

_REGISTRY: dict[str, ModelKind] = {}


def register_model_kind(kind: ModelKind) -> ModelKind:
    """Register *kind* under its own ``name``, replacing any kind of that name."""
    _REGISTRY[kind.name] = kind
    return kind


def model_kind(
    name: str,
    *,
    config_keys: "frozenset[str] | set[str] | None" = None,
    accepts_global_instruction: bool = True,
):
    """Decorator form of :func:`register_model_kind`, wrapping an ``exec_fn``.

    ::

        @pbt.model_kind("shout", config_keys={"suffix"})
        async def shout(rendered, call):
            return (await call.llm(rendered)).upper()

    The function is returned unchanged, so it stays directly callable and
    testable on its own.
    """
    def decorator(fn: ExecFn) -> ExecFn:
        register_model_kind(ModelKind(
            name=name,
            exec_fn=fn,
            config_keys=frozenset(config_keys or ()),
            accepts_global_instruction=accepts_global_instruction,
        ))
        return fn

    return decorator


def get_model_kind(name: str) -> ModelKind | None:
    """Return the kind registered under *name*, or None if there is none."""
    return _REGISTRY.get(name)


def known_model_kinds() -> set[str]:
    """Return every registered model_type value, excluding the unnamed default."""
    return {name for name in _REGISTRY if name != DEFAULT_MODEL_TYPE}


def registered_config_keys() -> set[str]:
    """Return every config() key declared by a registered model kind."""
    keys: set[str] = set()
    for kind in _REGISTRY.values():
        keys.update(kind.config_keys)
    return keys
