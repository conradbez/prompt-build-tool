"""
Prompt executor — orchestrates the full run lifecycle.

For each model (in dependency order):
  1. Render the Jinja2 template, injecting upstream outputs via ref().
  2. Send the rendered prompt to Gemini.
  3. Persist input + output to SQLite.

Gemini configuration
--------------------
Set the GEMINI_API_KEY environment variable before running pbt.
The model can be overridden with GEMINI_MODEL (default: gemini-2.0-flash).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from pbt import db
from pbt.graph import PromptModel
from pbt.llm import resolve_llm_call
from pbt.rag import resolve_rag_call
from pbt.parser import render_prompt


@dataclass
class ModelRunResult:
    model_name: str
    status: str            # 'success' | 'error' | 'skipped'
    prompt_rendered: str = ""
    llm_output: str = ""
    error: str = ""
    execution_ms: int = 0
    cached: bool = False



def execute_run(
    run_id: str,
    ordered_models: list[PromptModel],
    models_dir: str = ".",
    preloaded_outputs: dict[str, str] | None = None,
    on_model_start: Callable[[str], None] | None = None,
    on_model_done: Callable[[ModelRunResult], None] | None = None,
    llm_call: Callable[[str], str] | None = None,
    rag_call: Callable[..., list] | None = None,
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
    on_model_start / on_model_done:
        Optional progress callbacks for the CLI layer.

    Returns
    -------
    List of ModelRunResult, one per model.
    """
    if llm_call is None:
        llm_call = resolve_llm_call(models_dir)
    if rag_call is None:
        rag_call = resolve_rag_call(models_dir)

    # Seed model_outputs with any preloaded results from a previous run.
    model_outputs: dict[str, str] = dict(preloaded_outputs or {})

    # Register all models as 'pending' up front (mirrors dbt's deferred state).
    for model in ordered_models:
        db.upsert_model_pending(
            run_id=run_id,
            model_name=model.name,
            prompt_template=model.source,
            depends_on=model.depends_on,
        )

    results: list[ModelRunResult] = []
    failed_upstream: set[str] = set()

    for model in ordered_models:
        # Skip if any dependency failed *in this run* (preloaded deps are fine)
        blocked_by = [d for d in model.depends_on if d in failed_upstream]
        if blocked_by:
            db.mark_model_skipped(run_id, model.name)
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

        if on_model_start:
            on_model_start(model.name)

        db.mark_model_running(run_id, model.name)

        try:
            rendered = render_prompt(model.source, model_outputs, rag_call=rag_call)

            cached = db.get_cached_llm_output(rendered)
            if cached is not None:
                llm_output = cached
                elapsed_ms = 0
            else:
                t0 = time.monotonic()
                llm_output = llm_call(rendered)
                elapsed_ms = int((time.monotonic() - t0) * 1000)

            model_outputs[model.name] = llm_output
            db.mark_model_success(run_id, model.name, rendered, llm_output)

            result = ModelRunResult(
                model_name=model.name,
                status="success",
                prompt_rendered=rendered,
                llm_output=llm_output,
                execution_ms=elapsed_ms,
                cached=cached is not None,
            )

        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            db.mark_model_error(run_id, model.name, error_msg)
            failed_upstream.add(model.name)
            result = ModelRunResult(
                model_name=model.name,
                status="error",
                error=error_msg,
            )

        results.append(result)
        if on_model_done:
            on_model_done(result)

    return results
