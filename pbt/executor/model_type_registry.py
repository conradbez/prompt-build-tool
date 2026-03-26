"""
Registry for model-type callbacks.

Two callback types are supported:

replace_node_in_dag
    Receives one owned node and the full (read-only) models dict and returns a
    list of PromptModel objects to be substituted in the DAG in place of that
    node.

    Design constraint: callbacks may only insert *new* nodes owned by that
    model type — they may NOT modify any other existing node in the DAG.  The
    owned node is replaced wholesale; its name may appear in the returned list
    (e.g. as a terminal pass-through) or be omitted entirely.

execute_node
    Receives the model plus the full executor context and returns a
    ModelRunResult.  Used for model types that run differently at execution
    time without transforming the DAG (e.g. ``loop``, ``execute_python``).

    Signature::

        async def handler(
            model, model_outputs, model_files, storage_backend,
            run_id, llm_call, rag_call, promptdata,
            prompt_skipped_models, parse_json_output,
        ) -> ModelRunResult
"""

from __future__ import annotations

from typing import Callable

from pbt.executor.graph import PromptModel

# Maps model_type string → replace_node_in_dag callback.
# Signature: (owned_model, all_models_readonly) -> list[PromptModel]
_REPLACE_NODE_CALLBACKS: dict[
    str,
    Callable[[PromptModel, dict[str, PromptModel]], list[PromptModel]],
] = {}

# Maps model_type string → execute_node callback.
# See module docstring for the expected async signature.
_EXECUTE_NODE_CALLBACKS: dict[str, Callable] = {}


def register_replace_node_callback(
    model_type: str,
    callback: Callable[[PromptModel, dict[str, PromptModel]], list[PromptModel]],
) -> None:
    """Register a ``replace_node_in_dag`` callback for *model_type*."""
    _REPLACE_NODE_CALLBACKS[model_type] = callback


def register_execute_node_callback(model_type: str, callback: Callable) -> None:
    """Register an ``execute_node`` callback for *model_type*.

    The callback must be an async function with the signature described in the
    module docstring.
    """
    _EXECUTE_NODE_CALLBACKS[model_type] = callback


def get_execute_node_callback(model_type: str) -> Callable | None:
    """Return the ``execute_node`` callback for *model_type*, or ``None``."""
    return _EXECUTE_NODE_CALLBACKS.get(model_type)


def apply_replace_node_callbacks(
    models: dict[str, PromptModel],
) -> dict[str, PromptModel]:
    """
    For each model whose ``model_type`` has a registered ``replace_node_in_dag``
    callback, call the callback and replace that node in the DAG with the
    returned list of nodes.

    Callbacks may only insert new nodes owned by that model type — they may
    NOT modify any other existing node in the DAG.  The owned node is replaced
    wholesale by the returned list.
    """
    result = dict(models)
    for model in list(models.values()):
        model_type = model.config.get("model_type")
        if model_type and model_type in _REPLACE_NODE_CALLBACKS:
            replacement_nodes = _REPLACE_NODE_CALLBACKS[model_type](model, models)
            del result[model.name]
            for node in replacement_nodes:
                result[node.name] = node
    return result
