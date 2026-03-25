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

import inspect
import json
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from pbt.executor.graph import PromptModel
from pbt.executor.model_constructs import execute_loop_model
from pbt.storage.base import StorageBackend
from pbt.executor.parser import render_prompt
from pbt.types import PromptFile
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
        preview = stripped[:120] + ("..." if len(stripped) > 120 else "")
        raise ValueError(
            f"output_format='json' set but response is not valid JSON "
            f"(at line {exc.lineno}, col {exc.colno}): {exc.msg}\n"
            f"Got: {preview!r}"
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
    prompt_skipped: bool = False  # True when a skip function fired during rendering



async def execute_run(
    run_id: str,
    ordered_models: list[PromptModel],
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

                if model.config.get("model_type") == "loop":
                    result = await execute_loop_model(
                        model=model,
                        model_outputs=model_outputs,
                        model_files=model_files,
                        storage_backend=storage_backend,
                        run_id=run_id,
                        llm_call=llm_call,
                        rag_call=rag_call,
                        promptdata=promptdata,
                        prompt_skipped_models=prompt_skipped_models,
                        parse_json_output=_parse_json_output,
                    )

                else:
                    # --- Normal (non-loop) model ---
                    rendered, skip_state = render_prompt(model.source, model_outputs, promptdata=promptdata, rag_call=rag_call, prompt_skipped_models=prompt_skipped_models)

                    # Cache key includes config so a config-only change (e.g. adding
                    # output_format: json) correctly busts the cache.
                    cache_key = rendered + "\x00" + json.dumps(model.config, sort_keys=True)

                    output_format = model.config.get("output_format", "text")
                    cached = None

                    if skip_state.skip_value is not None:
                        llm_output = skip_state.skip_value
                        model_outputs[model.name] = llm_output
                        elapsed_ms = 0
                        prompt_skipped_models.add(model.name)
                        if skip_state.skip_downstream:
                            skip_downstream_models.add(model.name)

                    elif (cached := storage_backend.get_cached_llm_output(cache_key)) is not None:
                        # Cached path: apply JSON parsing + validator once.
                        llm_output = cached
                        elapsed_ms = 0
                        if output_format == "json":
                            parsed = _parse_json_output(llm_output)
                            model_outputs[model.name] = parsed
                            llm_output = json.dumps(parsed)
                        else:
                            model_outputs[model.name] = llm_output
                        if validators:
                            validated = run_validator(model.name, validators, rendered, llm_output)
                            if isinstance(validated, (dict, list)):
                                model_outputs[model.name] = validated
                                llm_output = json.dumps(validated)
                            else:
                                llm_output = validated if isinstance(validated, str) else str(validated)
                                model_outputs[model.name] = llm_output

                    else:
                        # LLM call path — retry up to max_retries times if validator fails.
                        # JSON parsing and validation run inside the loop so any failure
                        # (bad JSON *or* False validator) triggers a fresh LLM call.
                        max_retries = int(model.config.get("max_retries", 1))
                        _sig = inspect.signature(llm_call).parameters
                        _kwargs: dict = {}
                        if model_files and "files" in _sig:
                            _kwargs["files"] = model_files
                        if "config" in _sig:
                            _kwargs["config"] = model.config

                        llm_output = ""
                        elapsed_ms = 0
                        _last_error: Exception | None = None

                        for _attempt in range(max_retries):
                            t0 = time.monotonic()
                            _raw = llm_call(rendered, **_kwargs)
                            llm_output = await _raw if inspect.isawaitable(_raw) else _raw
                            elapsed_ms += int((time.monotonic() - t0) * 1000)

                            try:
                                if output_format == "json":
                                    parsed = _parse_json_output(llm_output)
                                    model_outputs[model.name] = parsed
                                    llm_output = json.dumps(parsed)
                                else:
                                    model_outputs[model.name] = llm_output

                                if validators:
                                    validated = run_validator(model.name, validators, rendered, llm_output)
                                    if isinstance(validated, (dict, list)):
                                        model_outputs[model.name] = validated
                                        llm_output = json.dumps(validated)
                                    else:
                                        llm_output = validated if isinstance(validated, str) else str(validated)
                                        model_outputs[model.name] = llm_output

                                _last_error = None
                                break  # success — exit retry loop
                            except Exception as exc:
                                _last_error = exc  # will retry on next attempt

                        if _last_error is not None:
                            raise _last_error

                    storage_backend.mark_model_success(run_id, model.name, rendered, llm_output, cache_key=cache_key)

                    result = ModelRunResult(
                        model_name=model.name,
                        status="success",
                        prompt_rendered=rendered,
                        llm_output=llm_output,
                        execution_ms=elapsed_ms,
                        cached=cached is not None,
                        prompt_skipped=skip_state.skip_value is not None,
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
