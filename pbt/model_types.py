"""
Model-type registry — the single public place a new execution strategy is added.

A *model type* is the value of ``{{ config(model_type="...") }}`` in a .prompt
file.  It selects the strategy used to turn a rendered template into an output.
Everything that surrounds that step — rendering, the prompt cache, JSON parsing,
validation, storage, skip propagation — is owned by the executor and is the same
for every type (see :mod:`pbt.executor.executor`).

A strategy therefore only implements the interesting middle::

    import pbt

    @pbt.model_type("shout", config_keys={"volume"})
    class Shout:
        async def execute(self, spec, ctx):
            rendered, state = ctx.render(spec)
            return await ctx.call_llm(rendered, spec, state)

Registering also declares the ``config()`` keys the type consumes, so they stop
triggering :class:`~pbt.executor.parser_initial.UnknownConfigKeyWarning` — there
is no second registry to keep in sync.

Where to register
-----------------
Anywhere that runs before the models are loaded.  ``client.py`` is the natural
home for a project-local type: pbt imports it before building the DAG precisely
so that types registered there are available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pbt.executor.run_context import RunContext
    from pbt.model_spec import ModelSpec


@runtime_checkable
class ModelType(Protocol):
    """Execution strategy for one ``model_type`` value.

    Only :meth:`execute` is required.  The class attributes and :meth:`expand`
    have defaults supplied by :class:`BaseModelType`, which most implementations
    should subclass rather than satisfying this protocol by hand.
    """

    #: config() keys this type consumes, beyond the ones pbt always knows.
    config_keys: frozenset[str]

    #: Whether the run's global instruction may be rendered into this model's
    #: prompt.  False for types whose rendered template is not natural language
    #: (Python source, a pass-through value) and so would be corrupted by it.
    accepts_global_instruction: bool

    def expand(
        self, spec: "ModelSpec", all_specs: "dict[str, ModelSpec]"
    ) -> "list[ModelSpec] | None":
        """Optionally rewrite this node into several nodes at DAG-build time."""
        ...

    async def execute(self, spec: "ModelSpec", ctx: "RunContext") -> Any:
        """Produce this model's output value."""
        ...


class BaseModelType:
    """Convenience base class supplying the optional half of :class:`ModelType`."""

    config_keys: frozenset[str] = frozenset()
    accepts_global_instruction: bool = True

    def expand(
        self, spec: "ModelSpec", all_specs: "dict[str, ModelSpec]"
    ) -> "list[ModelSpec] | None":
        """Return replacement nodes for *spec*, or None to leave it alone.

        Called once per model immediately after it is parsed, before the DAG is
        built.  ``all_specs`` holds the models parsed so far and is read-only —
        a type may only describe itself and nodes it invents.

        The returned list replaces *spec* entirely and **must** contain exactly
        one spec named ``spec.name``, so that downstream ``ref()`` calls keep
        resolving.  This is the "one node in, N nodes out" expansion used by
        ``quality_check``.
        """
        return None

    async def execute(self, spec: "ModelSpec", ctx: "RunContext") -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "'async def execute(self, spec, ctx)'."
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ModelType] = {}

#: model_type value for a plain LLM call — what a .prompt file gets when it
#: sets no model_type at all.
DEFAULT_MODEL_TYPE = ""

_builtins_loaded = False


def _ensure_builtins() -> None:
    """Import the built-in strategies so they self-register.

    Deferred because the built-ins import the renderer and the run context,
    while this module is imported very early (by the config parser).
    """
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    import pbt.executor.builtin_types  # noqa: F401  (registers on import)


def register_model_type(
    name: str,
    model_type: "ModelType | type[ModelType]",
    *,
    config_keys: "frozenset[str] | set[str] | None" = None,
) -> "ModelType":
    """Register *model_type* under the ``config(model_type=name)`` value *name*.

    Accepts either a class (instantiated with no arguments) or a ready
    instance — a strategy holds no per-model state, so one instance serves
    every model of that type.

    *config_keys* adds to whatever the class declares, for the case where the
    keys are easier to state at the call site than on the class.
    """
    _ensure_builtins()
    instance = model_type() if isinstance(model_type, type) else model_type
    if config_keys:
        instance.config_keys = frozenset(instance.config_keys) | frozenset(config_keys)
    _REGISTRY[name] = instance
    return instance


def model_type(
    name: str, *, config_keys: "frozenset[str] | set[str] | None" = None
):
    """Class decorator form of :func:`register_model_type`.

    ::

        @pbt.model_type("shout", config_keys={"volume"})
        class Shout(pbt.BaseModelType):
            async def execute(self, spec, ctx): ...
    """
    def decorator(cls):
        register_model_type(name, cls, config_keys=config_keys)
        return cls

    return decorator


def get_model_type(name: str) -> "ModelType | None":
    """Return the strategy registered under *name*, or None if there is none."""
    _ensure_builtins()
    return _REGISTRY.get(name)


def known_model_types() -> set[str]:
    """Return every registered model_type value, excluding the unnamed default."""
    _ensure_builtins()
    return {name for name in _REGISTRY if name != DEFAULT_MODEL_TYPE}


def registered_config_keys() -> set[str]:
    """Return every config() key declared by a registered model type."""
    _ensure_builtins()
    keys: set[str] = set()
    for instance in _REGISTRY.values():
        keys.update(instance.config_keys)
    return keys
