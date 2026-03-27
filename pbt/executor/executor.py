"""
Prompt executor — orchestrates the full run lifecycle.

For each model (in dependency order):
  1. Render the Jinja2 template, injecting upstream outputs via ref().
  2. Send the rendered prompt to the LLM.
  3. Persist input + output to storage.

LLM configuration
-----------------
Use ``pbt.llm.resolve_llm_call(models_dir)`` to auto-discover from client.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from pbt.executor.model_constructs import BaseModelHandler
from pbt.storage.base import StorageBackend
from pbt.types import PromptFile


@dataclass
class ModelRunResult:
    model_name: str
    status: str            # 'success' | 'error' | 'skipped'
    prompt_rendered: str = ""
    llm_output: str = ""
    error: str = ""
    execution_ms: int = 0
    cached: bool = False
    prompt_skipped: bool = False  # True when a skip function fired during rendering


async def execute_run(
    run_id: str,
    ordered_models: list[BaseModelHandler],
    storage_backend: StorageBackend,
    preloaded_outputs: dict[str, str] | None = None,
    on_model_start: Callable[[str], None] | None = None,
    on_model_done: Callable[[ModelRunResult], None] | None = None,
    llm_call: Callable[[str], str | Awaitable[str]] | None = None,
    rag_call: Callable[..., list] | None = None,
    promptdata: dict | None = None,
    promptfiles: dict[str, PromptFile] | None = None,
    validators: dict | None = None,
) -> list[ModelRunResult]:
    """
    Execute all *ordered_models* in sequence (dependency order).

    Parameters
    ----------
    run_id:
        The run ID created by db.create_run().
    ordered_models:
        Models sorted by execution_order() — upstream models first.
    preloaded_outputs:
        Outputs from a previous run to seed ref() lookups.  Used by
        ``--select`` so upstream models don't need to be re-executed.
    llm_call:
        LLM backend callable ``(prompt: str) -> str``. Required.
        Use ``pbt.llm.resolve_llm_call(models_dir)`` to auto-discover from client.py.
    rag_call:
        RAG backend callable or None.
    on_model_start / on_model_done:
        Optional progress callbacks for the CLI layer.

    Returns
    -------
    List of ModelRunResult, one per model.
    """
    if llm_call is None:
        raise ValueError(
            "llm_call must be provided to execute_run(). "
            "Use pbt.llm.resolve_llm_call(models_dir) to auto-discover from client.py."
        )

    # Seed model_outputs with any preloaded results from a previous run.
    model_outputs: dict[str, str] = dict(preloaded_outputs or {})
    # Tracks models whose LLM call was skipped via a skip function in the template.
    prompt_skipped_models: set[str] = set()
    # Tracks models that triggered skip_this_and_downstream — their dependents skip too.
    skip_downstream_models: set[str] = set()

    # Register all models as 'pending' up front (mirrors dbt's deferred state).
    for model in ordered_models:
        storage_backend.upsert_model_pending(
            run_id=run_id,
            model_name=model.name,
            prompt_template=model.source,
            depends_on=model.depends_on,
        )

    results: list[ModelRunResult] = []
    failed_upstream: set[str] = set()
    completed: set[str] = set(model_outputs)  # preloaded outputs count as completed

    pending = list(ordered_models)
    while pending:
        still_waiting = []
        made_progress = False

        for model in pending:
            # Deps still running — come back to this model next iteration
            waiting_deps = [d for d in model.depends_on if d not in completed and d not in failed_upstream]
            if waiting_deps:
                still_waiting.append(model)
                continue

            made_progress = True

            # Skip if any dependency failed *in this run* (preloaded deps are fine)
            blocked_by = [d for d in model.depends_on if d in failed_upstream]
            if blocked_by:
                storage_backend.mark_model_skipped(run_id, model.name)
                result = ModelRunResult(
                    model_name=model.name,
                    status="skipped",
                    error=f"Skipped because upstream models failed: {blocked_by}",
                )
                results.append(result)
                failed_upstream.add(model.name)
                if on_model_done:
                    on_model_done(result)
                continue

            # Skip if any dependency called skip_this_and_downstream
            skip_signalled_by = [d for d in model.depends_on if d in skip_downstream_models]
            if skip_signalled_by:
                storage_backend.mark_model_skipped(run_id, model.name)
                result = ModelRunResult(
                    model_name=model.name,
                    status="skipped",
                    error=f"Skipped because upstream models signalled skip_this_and_downstream: {skip_signalled_by}",
                )
                results.append(result)
                skip_downstream_models.add(model.name)  # propagate further downstream
                completed.add(model.name)
                if on_model_done:
                    on_model_done(result)
                continue

            if on_model_start:
                on_model_start(model.name)

            storage_backend.mark_model_running(run_id, model.name)

            try:
                # Resolve file paths declared in this model's config block
                model_files: list | None = None
                if model.promptfiles_used and promptfiles:
                    model_files = []
                    for name in model.promptfiles_used:
                        if name not in promptfiles:
                            raise ValueError(
                                f"Model '{model.name}' declares promptfile '{name}' in config "
                                f"but it was not provided. Pass it via --promptfile {name}=path or "
                                f"the promptfiles= argument."
                            )
                        pf = promptfiles[name]
                        # Open string/Path values so LLM clients always receive file objects
                        if isinstance(pf, (str, Path)):
                            pf = open(pf, "rb")  # noqa: WPS515
                        model_files.append(pf)

                result = await model.execute_node(
                    model_outputs=model_outputs,
                    model_files=model_files,
                    storage_backend=storage_backend,
                    run_id=run_id,
                    llm_call=llm_call,
                    rag_call=rag_call,
                    promptdata=promptdata,
                    prompt_skipped_models=prompt_skipped_models,
                    skip_downstream_models=skip_downstream_models,
                    validators=validators,
                )
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                storage_backend.mark_model_error(run_id, model.name, error_msg)
                failed_upstream.add(model.name)
                result = ModelRunResult(
                    model_name=model.name,
                    status="error",
                    error=error_msg,
                )
            else:
                completed.add(model.name)

            results.append(result)
            if on_model_done:
                on_model_done(result)

        pending = still_waiting
        if not made_progress:
            # No model could run this pass — unresolvable (e.g. circular deps).
            # Emit an error result for each stuck model so nothing is silently dropped.
            for model in still_waiting:
                storage_backend.mark_model_error(run_id, model.name, "Unresolvable dependency (possible cycle)")
                result = ModelRunResult(
                    model_name=model.name,
                    status="error",
                    error=f"Unresolvable dependency (possible cycle): {model.depends_on}",
                )
                results.append(result)
                if on_model_done:
                    on_model_done(result)
            break

    return results
