"""
Prompt executor — orchestrates the full run lifecycle.

For each model (in dependency order):
  1. Render the Jinja2 template, injecting upstream outputs via ref().
  2. Send the rendered prompt to Gemini.
  3. Persist input + output to SQLite.

Gemini configuration
--------------------
Set the GEMINI_API_KEY environment variable before running pbt.
The model can be overridden with GEMINI_MODEL (default: gemini-3-flash-preview).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable

from pbt import db
from pbt.executor.graph import PromptModel
from pbt.types import PromptFile
from pbt.executor.parser import render_prompt, SKIP_SENTINEL, _SKIP_OUTPUT, SKIP_AND_SET_PREFIX
from pbt.validator import run_validator

_JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _parse_json_output(raw: str) -> dict | list:
    """Strip optional ```json fences and parse as JSON. Raises ValueError on failure."""
    stripped = raw.strip()
    m = _JSON_FENCE.match(stripped)
    if m:
        stripped = m.group(1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model declared output_format: json but LLM returned invalid JSON: {exc}\n"
            f"Output was: {raw!r}"
        ) from exc


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
    preloaded_outputs: dict[str, str] | None = None,
    on_model_start: Callable[[str], None] | None = None,
    on_model_done: Callable[[ModelRunResult], None] | None = None,
    llm_call: Callable[[str], str] | None = None,
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
            rendered = render_prompt(model.source, model_outputs, promptdata=promptdata, rag_call=rag_call)

            # Resolve file paths declared in this model's config block
            model_files: list[str] | None = None
            if model.promptfiles_used and promptfiles:
                model_files = []
                for name in model.promptfiles_used:
                    if name not in promptfiles:
                        raise ValueError(
                            f"Model '{model.name}' declares promptfile '{name}' in config "
                            f"but it was not provided. Pass it via --promptfile {name}=path or "
                            f"the promptfiles= argument."
                        )
                    model_files.append(promptfiles[name])

            # Cache key includes config so a config-only change (e.g. adding
            # output_format: json) correctly busts the cache.
            cache_key = rendered + "\x00" + json.dumps(model.config, sort_keys=True)

            if rendered.strip() == SKIP_SENTINEL:
                llm_output = _SKIP_OUTPUT
                elapsed_ms = 0
            elif rendered.strip().startswith(SKIP_AND_SET_PREFIX):
                llm_output = rendered.strip()[len(SKIP_AND_SET_PREFIX):]
                elapsed_ms = 0
            elif (cached := db.get_cached_llm_output(cache_key)) is not None:
                llm_output = cached
                elapsed_ms = 0
            else:
                import inspect as _inspect
                t0 = time.monotonic()
                _sig = _inspect.signature(llm_call).parameters
                _kwargs: dict = {}
                if model_files and "files" in _sig:
                    _kwargs["files"] = model_files
                if "config" in _sig:
                    _kwargs["config"] = model.config
                llm_output = llm_call(rendered, **_kwargs)
                elapsed_ms = int((time.monotonic() - t0) * 1000)

            # If model declares output_format: json, validate and parse output.
            # Downstream ref() will receive a Python dict/list instead of a string.
            output_format = model.config.get("output_format", "text")
            if llm_output != _SKIP_OUTPUT and output_format == "json":
                parsed = _parse_json_output(llm_output)
                model_outputs[model.name] = parsed
                # Normalise to canonical JSON for DB storage
                llm_output = json.dumps(parsed)
            else:
                model_outputs[model.name] = llm_output

            # Run user-defined validator as a post-processing step.
            # Its return value becomes the model's output; False → error.
            if llm_output != _SKIP_OUTPUT and validators:
                validated = run_validator(model.name, validators, rendered, llm_output)
                if isinstance(validated, (dict, list)):
                    model_outputs[model.name] = validated
                    llm_output = json.dumps(validated)
                else:
                    llm_output = validated if isinstance(validated, str) else str(validated)
                    model_outputs[model.name] = llm_output

            db.mark_model_success(run_id, model.name, rendered, llm_output, cache_key=cache_key)

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
