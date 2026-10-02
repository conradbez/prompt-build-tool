"""
Prompt executor — orchestrates the full run lifecycle.

The executor owns everything that is the same for every model, so that a model
kind never has to reimplement it:

  1. Look up the kind for the model's ``model_type``.
  2. Render the template — once, or once per item for ``config(each=...)``.
  3. Hand the rendered text to the kind's ``exec_fn``, with the cached LLM call
     and cached compute preloaded onto a :class:`~pbt.model_types.ModelCall`.
  4. Apply skip propagation from the model's own template.
  5. Parse the value when ``output_format="json"``.
  6. Persist prompt and output, which is also what populates the prompt cache.
  7. Run the model's validator, if it has one.

Adding a model kind therefore means writing step 3 only, as one function — see
:mod:`pbt.model_types`.

LLM configuration
-----------------
Use ``pbt.llm.resolve_llm_call(models_dir)`` to auto-discover from client.py.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Awaitable, Callable

from pbt import jsonpath
from pbt.executor.parser_model import _RenderState
from pbt.executor.run_context import RunContext, parse_json_output
from pbt.files import BlobStore, contains_files, decode_output, encode_output, persist_files
from pbt.model_spec import ModelSpec
from pbt.model_types import ModelCall, ModelKind, get_model_kind
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
    #: Tokens spent by this model's calls, and those its cache hits saved.
    #: Unknown (None) unless ``llm_call`` returned a :class:`pbt.LLMResult`.
    spent_tokens: int | None = None
    cache_spent_tokens: int | None = None
    #: The output as downstream models see it — a parsed JSON value, or
    #: File/Dir/Output objects for a model that produced files.  ``llm_output``
    #: is its stored string form.
    value: Any = None


def _resolve_each(spec: ModelSpec, ctx: RunContext) -> tuple[str, list] | None:
    """``(path, items)`` for a model with ``config(each=...)``, else None."""
    if not spec.each:
        return None
    name, steps = jsonpath.parse(spec.each)
    label = f"Model '{spec.name}': each='{spec.each}'"
    value = jsonpath.resolve(ctx.outputs[name], steps, label)
    items = jsonpath.as_items(value)
    if items is None:
        raise ValueError(
            f"{label} does not return a JSON list, it is {jsonpath.describe(value)}. "
            "Point the path at a list (e.g. 'model.key[*]'), and make sure the "
            "upstream model has output_format='json'."
        )
    return name, items


def _store_item(
    spec: ModelSpec, ctx: RunContext, index: int, rendered: str, state: _RenderState, value: Any
) -> None:
    """Write one fan-out item as its own row, ``model[i]``.

    The row is the item's landmark in ``pbt docs`` and its prompt-cache entry:
    the raw response sits under the item's own cache key, so an unchanged item
    is served from cache on the next run while only changed items are re-sent.
    """
    name = f"{spec.name}[{index}]"
    ctx.storage.upsert_model_pending(
        ctx.run_id, name, spec.source, spec.depends_on, spec.model_type, spec.config
    )
    output = encode_output(value)
    cacheable = state.cache_artifact is not None and state.skip_value is None
    ctx.storage.mark_model_success(
        ctx.run_id,
        name,
        rendered,
        state.cache_artifact if cacheable else output,
        cache_key=state.cache_key if cacheable else None,
        cached=state.calls > 0 and state.cache_hits == state.calls,
    )
    record = getattr(ctx.storage, "record_validated_output", None)
    if cacheable and output != state.cache_artifact and record is not None:
        record(ctx.run_id, name, output)


#: A template whose whole body is one ref(), optionally after its config().
_SOLE_REF = re.compile(
    r"^\s*(?:\{\{\s*config\(.*?\)\s*\}\}\s*)?"
    r"\{\{\s*ref\(\s*(['\"])(?P<name>[^'\"]+)\1\s*\)\s*\}\}\s*$",
    re.DOTALL,
)


def _passthrough_files(spec: ModelSpec, ctx: RunContext) -> Any:
    """The upstream value a pure ``{{ ref('x') }}`` template forwards, if it has files.

    Rendering would turn files into their text handle.  A template that is
    nothing but one ref() — an alias model — forwards the files themselves instead.
    """
    match = _SOLE_REF.match(spec.source)
    if match is None:
        return None
    name, steps = jsonpath.parse(match.group("name"))
    if name not in ctx.outputs:
        return None
    value = jsonpath.resolve(ctx.outputs[name], steps, f"ref('{match.group('name')}')")
    return value if contains_files(value) else None


async def _produce_one(
    kind: ModelKind,
    spec: ModelSpec,
    ctx: RunContext,
    rendered: str,
    state: _RenderState,
) -> Any:
    """Run *kind*'s exec_fn over one rendered prompt.

    ``exec_fn=None`` means the rendered text is itself the output, so nothing
    runs.  Otherwise the cached LLM call and cached compute are bound to this
    model and this render — a kind receives them ready to call, and so never
    touches the cache, the clock or the skip state itself.
    """
    if kind.exec_fn is None:
        passthrough = _passthrough_files(spec, ctx)
        return rendered if passthrough is None else passthrough
    if state.skip_value is not None:
        return state.skip_value  # a skip function replaced the work itself

    # An each= item overlays the iterated model with the current item, for
    # code that reads call.outputs (Python ref()) just as for Jinja ref().
    overlay = state.extra_outputs
    call = ModelCall(
        spec=spec,
        outputs={**ctx.outputs, **overlay} if overlay else ctx.outputs,
        llm=partial(ctx.call_llm, spec=spec, state=state),
        compute=partial(ctx.cached, spec=spec, state=state),
    )
    return await kind.exec_fn(rendered, call)


async def _produce(kind: ModelKind, spec: ModelSpec, ctx: RunContext) -> Any:
    """Render *spec* and produce its output value.

    A model with ``each=`` renders once per item and runs
    its exec_fn on each concurrently, collecting the results in input order.
    Per-item renders are not *primary*: one skipped item must not mark the whole
    model skipped.
    """
    fan = _resolve_each(spec, ctx)
    if fan is None:
        rendered, state = ctx.render(spec)
        return await _produce_one(kind, spec, ctx, rendered, state)

    dep_name, items = fan
    ctx.note(spec, f"[each over {len(items)} items from '{spec.each}']")
    renders = [
        ctx.render(spec, extra_outputs={dep_name: item}, primary=False)
        for item in items
    ]

    async def one(rendered: str, state: _RenderState) -> Any:
        value = await _produce_one(kind, spec, ctx, rendered, state)
        if state.skip_value is None and isinstance(value, str) and spec.output_format == "json":
            return parse_json_output(value)
        return value

    values = list(await asyncio.gather(
        *(one(rendered, state) for rendered, state in renders)
    ))
    for index, ((rendered, state), value) in enumerate(zip(renders, values)):
        _store_item(spec, ctx, index, rendered, state, value)
    return values


async def execute_model(spec: ModelSpec, ctx: RunContext) -> ModelRunResult:
    """Run one model through the full lifecycle and return its result.

    The kind contributes the output value; every step around it is here, so
    caching, skipping, JSON handling, storage and validation behave identically
    for built-in and user-registered model kinds alike.
    """
    kind = get_model_kind(spec.model_type) or get_model_kind("")
    value = await _produce(kind, spec, ctx)
    # Files a kind built itself, outside the cached call, still need storing.
    persist_files(value, ctx.blobs)

    # --- skip propagation, from the model's own (primary) render -----------
    state = ctx.render_state(spec.name)
    skipped = state is not None and state.skip_value is not None
    if skipped:
        ctx.skipped.add(spec.name)
        if state.skip_downstream:
            ctx.skip_downstream.add(spec.name)

    # --- output_format ------------------------------------------------------
    # Only a plain string needs parsing: a kind that already produced a
    # structured value (a fan-out's list of items) has handled its own items.
    if not skipped and isinstance(value, str) and spec.output_format == "json":
        value = parse_json_output(value)

    ctx.outputs[spec.name] = value
    output = encode_output(value)
    rendered = ctx.prompt_rendered(spec.name)

    # --- persist ------------------------------------------------------------
    # What goes under the cache key is the raw LLM response, not this model's
    # final output.  The two differ whenever a kind post-processes the
    # response, and caching the processed form would re-apply that processing
    # on the next run.  Storing the raw form also means editing a validator
    # costs nothing at the LLM.
    cached_value = ctx.cache_artifact(spec.name)
    if cached_value is None:
        cached_value = output
    # Store under the key the call was looked up with.  A kind may widen it
    # beyond the rendered prompt (execute_python adds its upstream outputs).
    looked_up = state.cache_key if state is not None and state.calls == 1 else None
    ctx.storage.mark_model_success(
        ctx.run_id,
        spec.name,
        rendered,
        cached_value,
        cache_key=looked_up or ctx.cache_key(spec, rendered, ctx.files_for(spec)),
        cached=ctx.served_from_cache(spec.name),
    )
    spent, cache_spent = ctx.spent_tokens(spec.name), ctx.cache_spent_tokens(spec.name)
    record_tokens = getattr(ctx.storage, "record_token_usage", None)
    if record_tokens is not None and (spent is not None or cache_spent is not None):
        record_tokens(ctx.run_id, spec.name, spent_tokens=spent, cache_spent_tokens=cache_spent)

    # --- validate -----------------------------------------------------------
    if not skipped and ctx.validators:
        from pbt.validator import run_validator

        # A validator of a file-producing model gets the objects, not the
        # encoded string, so it can inspect the files themselves.
        given = value if contains_files(value) else output
        validated = run_validator(spec.name, ctx.validators, rendered, given)
        if contains_files(given) and (validated is given or validated is True):
            pass  # accepted as is
        elif contains_files(validated):
            persist_files(validated, ctx.blobs)
            ctx.outputs[spec.name] = validated
            output = encode_output(validated)
        elif isinstance(validated, (dict, list)):
            ctx.outputs[spec.name] = validated
            output = json.dumps(validated)
        else:
            text = validated if isinstance(validated, str) else str(validated)
            ctx.outputs[spec.name] = text
            output = encode_output(text)

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
        value=ctx.outputs[spec.name],
        spent_tokens=spent,
        cache_spent_tokens=cache_spent,
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
    blob_store: BlobStore | None = None,
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
        ``{{ config(global_instruction=False) }}``, and model kinds whose
        rendered template is not natural language never receive it.
    blob_store:
        Where file outputs' bytes are kept.  Defaults to the storage
        backend's own blob store (see :mod:`pbt.files`).

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
        blobs=blob_store,
    )
    for name, raw in (preloaded_outputs or {}).items():
        ctx.outputs[name] = decode_output(raw, ctx.blobs) if isinstance(raw, str) else raw

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
