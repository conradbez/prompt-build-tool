"""Base class for all prompt model types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar

if TYPE_CHECKING:
    from pbt.executor.executor import ModelRunResult


@dataclass
class BaseModelHandler:
    """Carries model data (source, config, deps) and execution logic.

    Subclasses override ``execute_node`` for custom model types.
    The default implementation (``NormalModelHandler``) handles plain LLM calls.
    """

    name: str
    path: Path
    source: str
    depends_on: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    promptdata_used: list[str] = field(default_factory=list)
    promptfiles_used: list[str] = field(default_factory=list)

    model_type: ClassVar[str] = ""

    async def execute_node(
        self,
        model_outputs: dict,
        model_files: list | None,
        storage_backend,
        run_id: str,
        llm_call: Callable,
        rag_call: Callable | None,
        promptdata: dict | None,
        prompt_skipped_models: set[str],
        skip_downstream_models: set[str],
        validators: dict | None = None,
    ) -> "ModelRunResult":
        raise NotImplementedError(
            f"{type(self).__name__} does not implement execute_node"
        )
