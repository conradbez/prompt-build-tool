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

import os
import time
from dataclasses import dataclass
from typing import Callable

from pbt import db
from pbt.graph import PromptModel
from pbt.parser import render_prompt

_DEFAULT_MODEL = "gemini-2.0-flash"


@dataclass
class ModelRunResult:
    model_name: str
    status: str            # 'success' | 'error' | 'skipped'
    prompt_rendered: str = ""
    llm_output: str = ""
    error: str = ""
    execution_ms: int = 0


def _gemini_client():
    # Lazy import so non-LLM commands (pbt ls, pbt show-runs, etc.) work even
    # if the google-generativeai package has environmental issues.
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is required to run prompts. "
            "Install it with: pip install google-generativeai"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Obtain a key at https://aistudio.google.com/app/apikey "
            "and export it:\n\n  export GEMINI_API_KEY=your_key_here"
        )
    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)
    return genai.GenerativeModel(model_name)


def execute_run(
    run_id: str,
    ordered_models: list[PromptModel],
    on_model_start: Callable[[str], None] | None = None,
    on_model_done: Callable[[ModelRunResult], None] | None = None,
) -> list[ModelRunResult]:
    """
    Execute all *ordered_models* in sequence (dependency order).

    Parameters
    ----------
    run_id:
        The run ID created by db.create_run().
    ordered_models:
        Models sorted by execution_order() — upstream models first.
    on_model_start / on_model_done:
        Optional progress callbacks for the CLI layer.

    Returns
    -------
    List of ModelRunResult, one per model.
    """
    client = _gemini_client()

    # Register all models as 'pending' up front (mirrors dbt's deferred state).
    for model in ordered_models:
        db.upsert_model_pending(
            run_id=run_id,
            model_name=model.name,
            prompt_template=model.source,
            depends_on=model.depends_on,
        )

    results: list[ModelRunResult] = []
    model_outputs: dict[str, str] = {}   # name → llm output (for ref())
    failed_upstream: set[str] = set()

    for model in ordered_models:
        # Skip if any dependency failed
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
            rendered = render_prompt(model.source, model_outputs)
            t0 = time.monotonic()
            response = client.generate_content(rendered)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            llm_output = response.text

            model_outputs[model.name] = llm_output
            db.mark_model_success(run_id, model.name, rendered, llm_output)

            result = ModelRunResult(
                model_name=model.name,
                status="success",
                prompt_rendered=rendered,
                llm_output=llm_output,
                execution_ms=elapsed_ms,
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
