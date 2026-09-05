"""
:class:`RunContext` — every run-wide input and every shared mechanism, in one
object that is threaded through the run instead of a dozen keyword arguments.

The executor owns it.  A model kind never sees it: the executor calls

``ctx.render(spec)``
    Render the model's template, resolving ``ref()`` against outputs produced
    so far and applying the run's global instruction.

and binds the two mechanisms a kind may need onto a
:class:`~pbt.model_types.ModelCall`, with the model and its render state
already applied:

``ctx.call_llm(rendered, spec, state)`` → ``call.llm(rendered)``
    Send a rendered prompt.  This is the *only* place that builds a cache key,
    consults the prompt cache, inspects ``llm_call``'s signature for optional
    ``files``/``config`` parameters, times the call, and copes with the callable
    being sync or async.

``ctx.cached(rendered, spec, state, compute=...)`` → ``call.compute(rendered, compute=...)``
    The same treatment for work that is not an LLM call.

Adding a new run-wide input is a field here, not a new parameter on six
functions.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pbt.executor.parser_model import _RenderState, render_prompt
from pbt.model_spec import ModelSpec
from pbt.storage.base import StorageBackend
from pbt.types import PromptFile

_JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

# config(global_instruction=...) values that turn the global instruction off.
_OPT_OUT_VALUES = frozenset({"false", "none", "no", "off", "0"})


def parse_json_output(raw: str) -> dict | list:
    """Strip optional ```json fences and parse as JSON. Raises ValueError on failure."""
    stripped = raw.strip()
    match = _JSON_FENCE.match(stripped)
    if match:
        stripped = match.group(1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        preview = stripped[:120] + ("..." if len(stripped) > 120 else "")
        raise ValueError(
            f"output_format='json' set but response is not valid JSON "
            f"(at line {exc.lineno}, col {exc.colno}): {exc.msg}\n"
            f"Got: {preview!r}"
        ) from exc


def _files_hash(model_files: list | None) -> str:
    """Return a short hash of the contents of all promptfiles, or '' if none."""
    if not model_files:
        return ""
    digest = hashlib.sha256()
    for handle in model_files:
        try:
            if isinstance(handle, (str, Path)):
                digest.update(Path(handle).read_bytes())
            else:
                data = handle.read()
                digest.update(data)
                handle.seek(0)
        except Exception:
            pass
    return digest.hexdigest()[:16]


@dataclass
class _ModelAccounting:
    """Per-model tallies the executor turns into a ModelRunResult."""

    rendered: list[str] = field(default_factory=list)
    note: str = ""
    elapsed_ms: int = 0
    calls: int = 0
    cache_hits: int = 0
    primary_state: _RenderState | None = None

    #: What ctx.cached() actually computed (or served) for this model — the raw
    #: LLM response, before any post-processing the strategy applies to it.
    #: This, not the strategy's return value, is what belongs under the cache
    #: key. None when the model made no calls, or more than one (a fan-out),
    #: where there is no single artifact to attribute.
    cache_artifact: str | None = None


@dataclass
class RunContext:
    """Run-wide inputs and shared mechanisms, owned by the executor.

    A model kind never sees this object: the executor binds the two pieces a
    kind may need — the cached LLM call and cached compute — onto a
    :class:`~pbt.model_types.ModelCall` for it.
    """

    run_id: str
    storage: StorageBackend
    llm_call: Callable[..., Any]
    rag_call: Callable[..., list] | None = None
    promptdata: dict | None = None
    promptfiles: dict[str, PromptFile] | None = None
    validators: dict | None = None
    global_instruction: str | None = None

    #: model name → output value.  A JSON-format model's entry is the parsed
    #: object, so downstream templates can do ``{{ ref('m').key }}``.
    outputs: dict[str, Any] = field(default_factory=dict)

    #: Models whose LLM call a skip function replaced during rendering.
    skipped: set[str] = field(default_factory=set)

    #: Models that called skip_this_and_downstream — their dependents skip too.
    skip_downstream: set[str] = field(default_factory=set)

    _accounting: dict[str, _ModelAccounting] = field(default_factory=dict)
    _files: dict[str, list | None] = field(default_factory=dict)

    # -- rendering ----------------------------------------------------------

    def render(
        self,
        spec: ModelSpec,
        extra_outputs: dict | None = None,
        primary: bool = True,
    ) -> tuple[str, _RenderState]:
        """Render *spec*'s template and record it for the run report.

        *extra_outputs* overlays the outputs used to resolve ``ref()`` — a loop
        uses it to make ``ref('items')`` yield the current item.

        *primary* marks the render whose skip functions govern the model as a
        whole.  Per-item renders inside a fan-out pass ``primary=False``, so one
        skipped item does not mark the whole model skipped.  Skip propagation
        itself is applied by the executor, from the primary state.
        """
        outputs = {**self.outputs, **extra_outputs} if extra_outputs else self.outputs
        rendered, state = render_prompt(
            spec.source,
            outputs,
            promptdata=self.promptdata,
            rag_call=self.rag_call,
            prompt_skipped_models=self.skipped,
            model_name=spec.name,
            global_instruction=self.global_instruction_for(spec),
        )
        acct = self._acct(spec.name)
        acct.rendered.append(rendered)
        if primary:
            acct.primary_state = state
        return rendered, state

    def global_instruction_for(self, spec: ModelSpec) -> str | None:
        """Return the global instruction to render into *spec*, or None.

        Suppressed when the model kind does not accept one (its rendered
        template is not natural language) or when the model opts out with
        ``{{ config(global_instruction=False) }}``.
        """
        if not self.global_instruction:
            return None
        from pbt.model_types import get_model_kind

        kind = get_model_kind(spec.model_type)
        if kind is not None and not kind.accepts_global_instruction:
            return None
        opt_out = str(spec.config.get("global_instruction", "")).strip().lower()
        if opt_out in _OPT_OUT_VALUES:
            return None
        return self.global_instruction

    def note(self, spec: ModelSpec, text: str) -> None:
        """Set a one-line header shown above a multi-prompt model's rendered text."""
        self._acct(spec.name).note = text

    # -- the LLM call -------------------------------------------------------

    def cache_key(self, spec: ModelSpec, rendered: str, files: list | None) -> str:
        """The prompt cache key: rendered prompt, config, and attached files."""
        return (
            rendered
            + "\x00"
            + json.dumps(spec.config, sort_keys=True)
            + "\x00"
            + _files_hash(files)
        )

    async def cached(
        self,
        rendered: str,
        spec: ModelSpec,
        state: _RenderState | None = None,
        *,
        compute: Callable[[], Any],
    ) -> str:
        """Return the memoised result of *compute* for this rendered prompt.

        The one caching path in pbt.  Anything expensive and deterministic that
        a kind derives from a rendered template goes through here, whether that
        is an LLM call or a block of Python — so every model kind gets the same
        cache, timing and skip behaviour.

        When *state* carries a skip value the work is bypassed entirely and that
        value is returned, so a kind never has to branch on skipping.
        *compute* may be sync or async.
        """
        if state is not None and state.skip_value is not None:
            return state.skip_value

        acct = self._acct(spec.name)
        key = self.cache_key(spec, rendered, self.files_for(spec))

        hit = self.storage.get_cached_llm_output(key)
        if hit is not None:
            acct.calls += 1
            acct.cache_hits += 1
            self._record_artifact(acct, hit)
            return hit

        started = time.monotonic()
        result = compute()
        if inspect.isawaitable(result):
            result = await result
        acct.calls += 1
        acct.elapsed_ms += int((time.monotonic() - started) * 1000)
        self._record_artifact(acct, result)
        return result

    @staticmethod
    def _record_artifact(acct: _ModelAccounting, result: str) -> None:
        """Remember a single call's result; forget it once there are several."""
        acct.cache_artifact = result if acct.calls == 1 else None

    async def call_llm(
        self,
        rendered: str,
        spec: ModelSpec,
        state: _RenderState | None = None,
    ) -> str:
        """Send *rendered* to the LLM backend, or serve it from the prompt cache.

        Optional ``files`` and ``config`` parameters are passed only when the
        backend's signature declares them, so a two-line ``llm_call`` stays
        valid.  A synchronous backend runs in a worker thread, which is what
        lets a fan-out model issue its calls concurrently.
        """
        return await self.cached(
            rendered, spec, state, compute=lambda: self._invoke_llm(rendered, spec)
        )

    async def _invoke_llm(self, rendered: str, spec: ModelSpec) -> str:
        kwargs: dict = {}
        params = inspect.signature(self.llm_call).parameters
        files = self.files_for(spec)
        if files and "files" in params:
            kwargs["files"] = files
        if "config" in params:
            kwargs["config"] = spec.config

        if inspect.iscoroutinefunction(self.llm_call):
            return await self.llm_call(rendered, **kwargs)

        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(
            None, lambda: self.llm_call(rendered, **kwargs)
        )
        if inspect.isawaitable(output):
            output = await output
        return output

    def files_for(self, spec: ModelSpec) -> list | None:
        """Open and return the promptfiles *spec* declares, or None.

        Opened once per model and reused, so every call for a fan-out model
        shares one set of handles.
        """
        if spec.name in self._files:
            return self._files[spec.name]

        if not spec.promptfiles_used or not self.promptfiles:
            self._files[spec.name] = None
            return None

        opened: list = []
        for name in spec.promptfiles_used:
            if name not in self.promptfiles:
                raise ValueError(
                    f"Model '{spec.name}' declares promptfile '{name}' in config "
                    f"but it was not provided. Pass it via --promptfile {name}=path "
                    "or the promptfiles= argument."
                )
            handle = self.promptfiles[name]
            if isinstance(handle, (str, Path)):
                handle = open(handle, "rb")  # noqa: WPS515 — closed by the consumer
            opened.append(handle)

        self._files[spec.name] = opened
        return opened

    # -- accounting, read by the executor -----------------------------------

    def _acct(self, name: str) -> _ModelAccounting:
        return self._accounting.setdefault(name, _ModelAccounting())

    def primary_state(self, name: str) -> _RenderState | None:
        return self._acct(name).primary_state

    def prompt_rendered(self, name: str) -> str:
        """The rendered prompt(s) for *name*, as one string for storage.

        A fan-out model rendered many prompts; they are joined so the run report
        and the docs page show exactly what was sent.
        """
        acct = self._acct(name)
        body = "\n---\n".join(acct.rendered)
        return f"{acct.note}\n{body}" if acct.note else body

    def elapsed_ms(self, name: str) -> int:
        return self._acct(name).elapsed_ms

    def cache_artifact(self, name: str) -> str | None:
        """The raw result of this model's single LLM/compute call, if it made one.

        The executor stores this under the cache key rather than the kind's
        return value.  A kind that post-processes the response (uppercasing
        it, appending to it) must not have that processing baked into the cached
        entry, or the next run would apply it a second time.
        """
        return self._acct(name).cache_artifact

    def served_from_cache(self, name: str) -> bool:
        """True when every LLM call this model made was a cache hit."""
        acct = self._acct(name)
        return acct.calls > 0 and acct.cache_hits == acct.calls
