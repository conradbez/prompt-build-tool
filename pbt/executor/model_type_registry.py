"""
Registry for model-type handlers.

Define a subclass of ``BaseModelHandler`` for each model type and register an
instance with ``register_handler()``.  The executor discovers handlers via
``get_handler()`` at run time.

Two extension points are available on each handler:

inject_nodes_before
    Called during DAG construction.  Return a list of replacement
    ``PromptModel`` objects to substitute for the owned node, or ``None``
    to leave the node unchanged.

execute_node
    Called during execution for any model whose ``model_type`` matches this
    handler.  Must be an ``async`` method that mutates *model_outputs* and
    returns a ``ModelRunResult``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from pbt.executor.graph import PromptModel

if TYPE_CHECKING:
    from pbt.executor.executor import ModelRunResult


class BaseModelHandler:
    """Base class for model-type handlers.

    Subclasses must set ``model_type`` and override at least one of
    ``inject_nodes_before`` or ``execute_node``.
    """

    model_type: str  # must be set by each subclass

    def inject_nodes_before(
        self,
        model: PromptModel,
        all_models: dict[str, PromptModel],
    ) -> list[PromptModel] | None:
        """Return replacement nodes, or ``None`` to leave this node unchanged.

        Design constraint: only return nodes owned by this handler's
        ``model_type`` — never modify existing nodes in the DAG.
        """
        return None

    async def execute_node(
        self,
        model: PromptModel,
        model_outputs: dict,
        model_files: list | None,
        storage_backend,
        run_id: str,
        llm_call: Callable,
        rag_call: Callable | None,
        promptdata: dict | None,
        prompt_skipped_models: set[str],
        parse_json_output: Callable,
    ) -> "ModelRunResult":
        """Execute this model and return a ``ModelRunResult``.

        Must mutate ``model_outputs[model.name]`` before returning.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement execute_node"
        )


# Ordered list of registered handlers.
_HANDLERS: list[BaseModelHandler] = []


def register_handler(handler: BaseModelHandler) -> None:
    """Add *handler* to the global registry."""
    _HANDLERS.append(handler)


def get_handler(model_type: str) -> BaseModelHandler | None:
    """Return the handler for *model_type*, or ``None`` if not registered."""
    return next((h for h in _HANDLERS if h.model_type == model_type), None)


def apply_replace_node_callbacks(
    models: dict[str, PromptModel],
) -> dict[str, PromptModel]:
    """Expand any model whose type has a ``inject_nodes_before`` override.

    For each model whose ``model_type`` has a registered handler that returns
    a non-``None`` value from ``inject_nodes_before``, the original node is
    removed and the returned nodes are inserted in its place.
    """
    result = dict(models)
    for model in list(models.values()):
        model_type = model.config.get("model_type")
        if not model_type:
            continue
        handler = get_handler(model_type)
        if handler is None:
            continue
        replacements = handler.inject_nodes_before(model, models)
        if replacements is not None:
            del result[model.name]
            for node in replacements:
                result[node.name] = node
    return result
