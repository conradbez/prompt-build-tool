"""
Prompt executor — orchestrates the full run lifecycle.

The executor owns everything that is the same for every model, so that a model
type never has to reimplement it:

  1. Look up the strategy for the model's ``model_type``.
  2. Ask it for an output value — the strategy renders, and calls the LLM or
     runs whatever it runs, through the :class:`~pbt.executor.run_context.RunContext`.
  3. Apply skip propagation from the model's own template.
  4. Parse the value when ``output_format="json"``.
  5. Persist prompt and output, which is also what populates the prompt cache.
  6. Run the model's validator, if it has one.

Adding a model type therefore means writing step 2 only — see
:mod:`pbt.model_types`.

LLM configuration
-----------------
Use ``pbt.llm.resolve_llm_call(models_dir)`` to auto-discover from client.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Awaitable, Callable

from pbt.executor.run_context import RunContext, parse_json_output
from pbt.model_spec import ModelSpec
from pbt.model_types import get_model_type
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


async def execute_model(spec: ModelSpec, ctx: RunContext) -> ModelRunResult:
    """Run one model through the full lifecycle and return its result.

    The strategy contributes the output value; every step around it is here, so
    caching, skipping, JSON handling, storage and validation behave identically
    for built-in and user-registered model types alike.
    """
    strategy = get_model_type(spec.model_type) or get_model_type("")
    value = await strategy.execute(spec, ctx)

    # --- skip propagation, from the model's own (primary) render -----------
    state = ctx.primary_state(spec.name)
    skipped = state is not None and state.skip_value is not None
    if skipped:
        ctx.skipped.add(spec.name)
        if state.skip_downstream:
            ctx.skip_downstream.add(spec.name)

    # --- output_format ------------------------------------------------------
    # Only a plain string needs parsing: a strategy that already produced a
    # structured value (a fan-out's list of items) has handled its own items.
    if not skipped and isinstance(value, str) and spec.output_format == "json":
        value = parse_json_output(value)

    ctx.outputs[spec.name] = value
    output = value if isinstance(value, str) else json.dumps(value)
    rendered = ctx.prompt_rendered(spec.name)

    # --- persist ------------------------------------------------------------
    # What goes under the cache key is the raw LLM response, not this model's
    # final output.  The two differ whenever a strategy post-processes the
    # response, and caching the processed form would re-apply that processing
    # on the next run.  Storing the raw form also means editing a validator
    # costs nothing at the LLM.
    cached_value = ctx.cache_artifact(spec.name)
    if cached_value is None:
        cached_value = output
    ctx.storage.mark_model_success(
        ctx.run_id,
        spec.name,
        rendered,
        cached_value,
        cache_key=ctx.cache_key(spec, rendered, ctx.files_for(spec)),
        cached=ctx.served_from_cache(spec.name),
    )

    # --- validate -----------------------------------------------------------
    if not skipped and ctx.validators:
        from pbt.validator import run_validator

        validated = run_validator(spec.name, ctx.validators, rendered, output)
        if isinstance(validated, (dict, list)):
            ctx.outputs[spec.name] = validated
            output = json.dumps(validated)
        else:
            output = validated if isinstance(validated, str) else str(validated)
            ctx.outputs[spec.name] = output

    if output != cached_value:
        # Record the model's actual output next to the cached raw response, so
        # `pbt test` and `pbt docs` show what the pipeline really passed on.
        record = getattr(ctx.storage, "record_validated_output", None)
        if record is not None:
            record(ctx.run_id, spec.name, output)

    return ModelRunResult(
        model_name=spec.name,
        status="success",
        prompt_rendered=rendered,
        llm_output=output,
        execution_ms=ctx.elapsed_ms(spec.name),
        cached=ctx.served_from_cache(spec.name),
        prompt_skipped=skipped,
    )


async def execute_run(
    run_id: str,
    ordered_models: list[ModelSpec],
    storage_backend: StorageBackend,
    preloaded_outputs: dict[str, str] | None = None,
    on_model_start: Callable[[str], None] | None = None,
    on_model_done: Callable[[ModelRunResult], None] | None = None,
    llm_call: Callable[[str], str | Awaitable[str]] | None = None,
    rag_call: Callable[..., list] | None = None,
    promptdata: dict | None = None,
    promptfiles: dict[str, PromptFile] | None = None,
    validators: dict | None = None,
    global_instruction: str | None = None,
) -> list[ModelRunResult]:
    """
    Execute all *ordered_models* in dependency order.

    Parameters
    ----------
    run_id:
        The run ID created by db.create_run().
    ordered_models:
        The :class:`~pbt.model_spec.ModelSpec` objects to run.  Execution order
        is derived from their dependencies, so the list order does not matter.
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
    global_instruction:
        Optional prompt text rendered into every model's prompt — pbt's
        analogue of dbt's query-comment.  Individual models opt out with
        ``{{ config(global_instruction=False) }}``, and model types whose
        rendered template is not natural language never receive it.

    Returns
    -------
    List of ModelRunResult, one per model.
    """
    if llm_call is None:
        raise ValueError(
            "llm_call must be provided to execute_run(). "
            "Use pbt.llm.resolve_llm_call(models_dir) to auto-discover from client.py."
        )

    ctx = RunContext(
        run_id=run_id,
        storage=storage_backend,
        llm_call=llm_call,
        rag_call=rag_call,
        promptdata=promptdata,
        promptfiles=promptfiles,
        validators=validators,
        global_instruction=global_instruction,
        outputs=dict(preloaded_outputs or {}),
    )

    # Register all models as 'pending' up front (mirrors dbt's deferred state).
    for spec in ordered_models:
        storage_backend.upsert_model_pending(
            run_id=run_id,
            model_name=spec.name,
            prompt_template=spec.source,
            depends_on=spec.depends_on,
            model_type=spec.model_type,
            config=spec.config,
        )

    results: list[ModelRunResult] = []
    failed_upstream: set[str] = set()
    completed: set[str] = set(ctx.outputs)  # preloaded outputs count as completed

    pending = list(ordered_models)
    while pending:
        still_waiting: list[ModelSpec] = []
        made_progress = False

        for spec in pending:
            # Deps still running — come back to this model next iteration
            waiting_deps = [
                dep for dep in spec.depends_on
                if dep not in completed and dep not in failed_upstream
            ]
            if waiting_deps:
                still_waiting.append(spec)
                continue

            made_progress = True

            # Skip if any dependency failed *in this run* (preloaded deps are fine)
            blocked_by = [dep for dep in spec.depends_on if dep in failed_upstream]
            if blocked_by:
                storage_backend.mark_model_skipped(run_id, spec.name)
                result = ModelRunResult(
                    model_name=spec.name,
                    status="skipped",
                    error=f"Skipped because upstream models failed: {blocked_by}",
                )
                results.append(result)
                failed_upstream.add(spec.name)
                if on_model_done:
                    on_model_done(result)
                continue

            # Skip if any dependency called skip_this_and_downstream
            skip_signalled_by = [
                dep for dep in spec.depends_on if dep in ctx.skip_downstream
            ]
            if skip_signalled_by:
                storage_backend.mark_model_skipped(run_id, spec.name)
                result = ModelRunResult(
                    model_name=spec.name,
                    status="skipped",
                    error=(
                        "Skipped because upstream models signalled "
                        f"skip_this_and_downstream: {skip_signalled_by}"
                    ),
                )
                results.append(result)
                ctx.skip_downstream.add(spec.name)  # propagate further downstream
                completed.add(spec.name)
                if on_model_done:
                    on_model_done(result)
                continue

            if on_model_start:
                on_model_start(spec.name)

            storage_backend.mark_model_running(run_id, spec.name)

            try:
                result = await execute_model(spec, ctx)
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                storage_backend.mark_model_error(run_id, spec.name, error_msg)
                failed_upstream.add(spec.name)
                result = ModelRunResult(
                    model_name=spec.name,
                    status="error",
                    error=error_msg,
                )
            else:
                completed.add(spec.name)

            results.append(result)
            if on_model_done:
                on_model_done(result)

        pending = still_waiting
        if not made_progress:
            # No model could run this pass — unresolvable (e.g. circular deps).
            # Emit an error result for each stuck model so nothing is silently dropped.
            for spec in still_waiting:
                storage_backend.mark_model_error(
                    run_id, spec.name, "Unresolvable dependency (possible cycle)"
                )
                result = ModelRunResult(
                    model_name=spec.name,
                    status="error",
                    error=f"Unresolvable dependency (possible cycle): {spec.depends_on}",
                )
                results.append(result)
                if on_model_done:
                    on_model_done(result)
            break

    return results
