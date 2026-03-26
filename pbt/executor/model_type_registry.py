"""
Registry for model-type handlers.

Define a subclass of ``BaseModelHandler`` for each model type and register an
instance with ``register_handler()``.  The executor discovers handlers via
``get_handler()`` at run time.

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

    Subclasses must set ``model_type`` and override ``execute_node``.
    """

    model_type: str  # must be set by each subclass

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
