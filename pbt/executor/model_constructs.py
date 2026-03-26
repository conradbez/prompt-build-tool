"""
Model handler base class and concrete model-type handlers for ``loop``,
``execute_python``, and ``quality_check``.

``BaseModelHandler`` provides the default (normal LLM-call) execution strategy.
Subclasses override ``execute_node`` for custom model types and may override
``inject_extra_nodes`` to expand a single node into a list of replacement nodes
at DAG-build time (before execution begins).
"""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import re
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar

from pbt.executor.parser_model import render_prompt

if TYPE_CHECKING:
    from pbt.executor.executor import ModelRunResult
    from pbt.storage.base import StorageBackend


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
class BaseModelHandler:
    """Carries model data (source, config, deps) and execution logic.

    The default ``execute_node`` performs a plain LLM call (the "normal" model
    type).  Subclasses override it for custom model types and set ``model_type``
    to the corresponding config key.
    """

    name: str
    path: Path
    source: str
    depends_on: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    promptdata_used: list[str] = field(default_factory=list)
    promptfiles_used: list[str] = field(default_factory=list)

    model_type: ClassVar[str] = ""

    def inject_extra_nodes(
        self, all_models: "dict[str, BaseModelHandler]"
    ) -> "tuple[BaseModelHandler, list[BaseModelHandler]] | None":
        """Optionally rewrite this node and inject additional preceding nodes.

        Called inline immediately after this handler is added to the models
        dict, before the DAG is built.  ``all_models`` is the partially-built
        models dict and may be used read-only to look up already-loaded models.

        Return value
        ------------
        ``None``
            Leave this node unchanged (default).
        ``(updated_self, extra_nodes)``
            *updated_self* replaces this node in the models dict (same name,
            updated source/deps/config).  *extra_nodes* are additional handlers
            added to the models dict alongside it.  The handler may only
            influence itself and new nodes — it must not modify any entry in
            *all_models* that already existed before it was called.
        """
        return None

    async def execute_node(
        self,
        model_outputs: dict,
        model_files: list | None,
        storage_backend: "StorageBackend",
        run_id: str,
        llm_call: Callable,
        rag_call: Callable | None,
        promptdata: dict | None,
        prompt_skipped_models: set[str],
        skip_downstream_models: set[str],
        validators: dict | None = None,
    ) -> "ModelRunResult":
        from pbt.executor.executor import ModelRunResult

        rendered, skip_state = render_prompt(self.source, model_outputs, promptdata=promptdata, rag_call=rag_call, prompt_skipped_models=prompt_skipped_models, model_name=self.name)
        cache_key = rendered + "\x00" + json.dumps(self.config, sort_keys=True)

        cached = None
        if skip_state.skip_value is not None:
            llm_output = skip_state.skip_value
            elapsed_ms = 0
            prompt_skipped_models.add(self.name)
            if skip_state.skip_downstream:
                skip_downstream_models.add(self.name)
        elif (cached := storage_backend.get_cached_llm_output(cache_key)) is not None:
            llm_output = cached
            elapsed_ms = 0
        else:
            t0 = time.monotonic()
            _sig = inspect.signature(llm_call).parameters
            _kwargs: dict = {}
            if model_files and "files" in _sig:
                _kwargs["files"] = model_files
            if "config" in _sig:
                _kwargs["config"] = self.config
            result = llm_call(rendered, **_kwargs)
            if inspect.isawaitable(result):
                llm_output = await result
            else:
                llm_output = result
            elapsed_ms = int((time.monotonic() - t0) * 1000)

        output_format = self.config.get("output_format", "text")
        if skip_state.skip_value is None and output_format == "json":
            parsed = _parse_json_output(llm_output)
            model_outputs[self.name] = parsed
            llm_output = json.dumps(parsed)
        else:
            model_outputs[self.name] = llm_output

        if skip_state.skip_value is None and validators:
            from pbt.validator import run_validator
            validated = run_validator(self.name, validators, rendered, llm_output)
            if isinstance(validated, (dict, list)):
                model_outputs[self.name] = validated
                llm_output = json.dumps(validated)
            else:
                llm_output = validated if isinstance(validated, str) else str(validated)
                model_outputs[self.name] = llm_output

        storage_backend.mark_model_success(run_id, self.name, rendered, llm_output, cache_key=cache_key)

        return ModelRunResult(
            model_name=self.name,
            status="success",
            prompt_rendered=rendered,
            llm_output=llm_output,
            execution_ms=elapsed_ms,
            cached=cached is not None,
            prompt_skipped=skip_state.skip_value is not None,
        )


class LoopModelHandler(BaseModelHandler):
    """Fan out over a JSON list from an upstream model, one LLM call per item."""

    model_type: str = "loop"  # type: ignore[assignment]

    async def execute_node(
        self,
        model_outputs: dict,
        model_files: list | None,
        storage_backend: "StorageBackend",
        run_id: str,
        llm_call: Callable,
        rag_call: Callable | None,
        promptdata: dict | None,
        prompt_skipped_models: set[str],
        skip_downstream_models: set[str],
        validators: dict | None = None,
    ) -> "ModelRunResult":
        from pbt.executor.executor import ModelRunResult

        # Find the single upstream dep that returns a JSON list.
        list_deps = {
            dep: model_outputs[dep]
            for dep in self.depends_on
            if dep in model_outputs and isinstance(model_outputs[dep], list)
        }
        if not list_deps:
            raise ValueError(
                f"Loop model '{self.name}': no upstream dependency returns a JSON list. "
                "Ensure an upstream model has output_format='json' and returns a list."
            )
        if len(list_deps) > 1:
            raise ValueError(
                f"Loop model '{self.name}': multiple dependencies return lists: {list(list_deps)}. "
                "Add loop_over='model_name' to config() to disambiguate."
            )
        list_dep_name, list_items = next(iter(list_deps.items()))
        output_format = self.config.get("output_format", "text")

        # Render all item prompts up front (fast, synchronous).
        rendered_items: list[tuple[str, object]] = []
        for item in list_items:
            item_outputs = {**model_outputs, list_dep_name: item}
            item_rendered, item_skip_state = render_prompt(
                self.source, item_outputs,
                promptdata=promptdata, rag_call=rag_call,
                prompt_skipped_models=prompt_skipped_models,
                model_name=self.name,
            )
            rendered_items.append((item_rendered, item_skip_state))

        async def _call_one(item_rendered: str, item_skip_state) -> tuple[str, int, bool]:
            item_cache_key = item_rendered + "\x00" + json.dumps(self.config, sort_keys=True)

            if item_skip_state.skip_value is not None:
                return item_skip_state.skip_value, 0, True

            cached = storage_backend.get_cached_llm_output(item_cache_key)
            if cached is not None:
                return cached, 0, True

            t0 = time.monotonic()
            _sig = inspect.signature(llm_call).parameters
            _kwargs: dict = {}
            if model_files and "files" in _sig:
                _kwargs["files"] = model_files
            if "config" in _sig:
                _kwargs["config"] = self.config

            if inspect.iscoroutinefunction(llm_call):
                result = await llm_call(item_rendered, **_kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: llm_call(item_rendered, **_kwargs))
            return result, int((time.monotonic() - t0) * 1000), False

        call_results = await asyncio.gather(
            *[_call_one(rendered, skip_state) for rendered, skip_state in rendered_items]
        )

        loop_results: list = []
        total_elapsed_ms = 0
        all_cached = True
        rendered_prompts = [r for r, _ in rendered_items]

        for (item_rendered, item_skip_state), (item_output, elapsed, cached) in zip(rendered_items, call_results):
            total_elapsed_ms += elapsed
            if not cached:
                all_cached = False
            if item_skip_state.skip_value is None and output_format == "json":
                item_output = _parse_json_output(item_output)
            loop_results.append(item_output)

        model_outputs[self.name] = loop_results
        llm_output = json.dumps(loop_results)
        combined_rendered = (
            f"[loop over {len(list_items)} items from '{list_dep_name}']\n"
            + "\n---\n".join(rendered_prompts)
        )
        combined_cache_key = combined_rendered + "\x00" + json.dumps(self.config, sort_keys=True)
        storage_backend.mark_model_success(
            run_id, self.name, combined_rendered, llm_output,
            cache_key=combined_cache_key,
        )
        return ModelRunResult(
            model_name=self.name,
            status="success",
            prompt_rendered=combined_rendered,
            llm_output=llm_output,
            execution_ms=total_elapsed_ms,
            cached=all_cached,
            prompt_skipped=False,
        )


class ExecutePythonModelHandler(BaseModelHandler):
    """Render the template as Python and run it with exec()."""

    model_type: str = "execute_python"  # type: ignore[assignment]

    async def execute_node(
        self,
        model_outputs: dict,
        model_files: list | None,
        storage_backend: "StorageBackend",
        run_id: str,
        llm_call: Callable,
        rag_call: Callable | None,
        promptdata: dict | None,
        prompt_skipped_models: set[str],
        skip_downstream_models: set[str],
        validators: dict | None = None,
    ) -> "ModelRunResult":
        """Render the template as Python code and exec() it.

        Output is captured from stdout (``print(...)`` calls).  Alternatively,
        assign to a variable named ``output`` and that value will be used if
        stdout is empty.  Upstream model outputs are available via a ``ref``
        callable and as entries in the ``model_outputs`` dict.
        """
        from pbt.executor.executor import ModelRunResult

        rendered, skip_state = render_prompt(
            self.source, model_outputs,
            promptdata=promptdata, rag_call=rag_call,
            prompt_skipped_models=prompt_skipped_models,
            model_name=self.name,
        )

        cache_key = rendered + "\x00" + json.dumps(self.config, sort_keys=True)

        if skip_state.skip_value is not None:
            llm_output = skip_state.skip_value
            prompt_skipped_models.add(self.name)
            model_outputs[self.name] = llm_output
            storage_backend.mark_model_success(run_id, self.name, rendered, llm_output, cache_key=cache_key)
            return ModelRunResult(
                model_name=self.name,
                status="success",
                prompt_rendered=rendered,
                llm_output=llm_output,
                execution_ms=0,
                cached=False,
                prompt_skipped=True,
            )

        cached = storage_backend.get_cached_llm_output(cache_key)
        if cached is not None:
            output_format = self.config.get("output_format", "text")
            if output_format == "json":
                model_outputs[self.name] = _parse_json_output(cached)
            else:
                model_outputs[self.name] = cached
            return ModelRunResult(
                model_name=self.name,
                status="success",
                prompt_rendered=rendered,
                llm_output=cached,
                execution_ms=0,
                cached=True,
                prompt_skipped=False,
            )

        t0 = time.monotonic()

        stdout_buf = io.StringIO()
        local_ns: dict = {"model_outputs": model_outputs, "ref": lambda name: model_outputs.get(name)}
        with redirect_stdout(stdout_buf):
            exec(compile(rendered, f"<{self.name}>", "exec"), local_ns)  # noqa: S102

        stdout_output = stdout_buf.getvalue()
        if stdout_output:
            llm_output = stdout_output.rstrip("\n")
        elif "output" in local_ns:
            raw = local_ns["output"]
            llm_output = json.dumps(raw) if isinstance(raw, (dict, list)) else str(raw)
        else:
            llm_output = ""

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        output_format = self.config.get("output_format", "text")
        if output_format == "json":
            parsed = _parse_json_output(llm_output)
            model_outputs[self.name] = parsed
            llm_output = json.dumps(parsed)
        else:
            model_outputs[self.name] = llm_output

        storage_backend.mark_model_success(run_id, self.name, rendered, llm_output, cache_key=cache_key)
        return ModelRunResult(
            model_name=self.name,
            status="success",
            prompt_rendered=rendered,
            llm_output=llm_output,
            execution_ms=elapsed_ms,
            cached=False,
            prompt_skipped=False,
        )


class QualityCheckModelHandler(BaseModelHandler):
    """Expand a quality-check node into a retry chain at DAG-build time.

    At graph-construction time ``inject_extra_nodes`` replaces the single
    quality-check node with an interleaved chain of check + retry nodes,
    plus a terminal pass-through node that keeps the original name.

    Example — given ``article → article_quality → summary`` where
    ``article_quality`` has ``model_type="quality_check"`` and
    ``quality_retries=2``, the expansion produces::

        article_quality_1  ← quality check on original article
        article_1          ← retry (skips if quality_1 passed)
        article_quality_2  ← quality check on article_1
        article_2          ← retry (skips if quality_2 passed)
        article_quality    ← terminal pass-through (keeps original name)

    The quality-check prompt (the node's source template) should reference the
    upstream target via ``ref('...')``.  Pass ``quality_pass_marker`` in config
    to change the substring the handler looks for in the quality check's output
    to decide whether to skip the retry (default: ``"PASS"``).
    """

    model_type: str = "quality_check"  # type: ignore[assignment]

    def inject_extra_nodes(
        self, all_models: "dict[str, BaseModelHandler]"
    ) -> "tuple[BaseModelHandler, list[BaseModelHandler]]":
        retries = int(self.config.get("quality_retries", "2"))
        pass_marker = self.config.get("quality_pass_marker", "PASS")

        # Identify the single upstream model being quality-checked.
        target_deps = [d for d in self.depends_on if d in all_models]
        if len(target_deps) != 1:
            raise ValueError(
                f"Quality check node '{self.name}' must depend on exactly one upstream "
                f"model; got depends_on={self.depends_on!r}."
            )
        target_name = target_deps[0]
        target_model = all_models[target_name]

        quality_config = {k: v for k, v in self.config.items() if k != "model_type"}
        target_config = {k: v for k, v in target_model.config.items() if k != "model_type"}

        extra_nodes: list[BaseModelHandler] = []
        prev_retry_name = target_name  # name of the most recent article-like node

        for i in range(1, retries + 1):
            check_name = f"{self.name}_{i}"
            retry_name = f"{target_name}_{i}"

            # --- quality-check sub-node ---
            if i == 1:
                check_source = self.source
                check_deps = list(self.depends_on)
            else:
                # Rewrite the quality template to ref the previous retry instead
                # of the original target.
                check_source = self.source.replace(
                    f"ref('{target_name}')", f"ref('{prev_retry_name}')"
                ).replace(
                    f'ref("{target_name}")', f'ref("{prev_retry_name}")'
                )
                check_deps = [prev_retry_name]

            extra_nodes.append(BaseModelHandler(
                name=check_name,
                path=self.path,
                source=check_source,
                depends_on=check_deps,
                config=quality_config,
                promptdata_used=self.promptdata_used,
                promptfiles_used=self.promptfiles_used,
            ))

            # --- retry sub-node ---
            # Skips (and passes through the previous best output) when quality passed.
            skip_block = (
                f"{{% if '{pass_marker}' in ref('{check_name}')|upper %}}"
                f"{{{{ skip_and_set_to_value(ref('{prev_retry_name}')) }}}}"
                f"{{% endif %}}\n"
            )
            feedback_block = (
                f"{{% set _ = model.meta._set(feedback_from_previous_run=ref('{check_name}')) %}}\n"
            )
            retry_source = skip_block + feedback_block + target_model.source
            retry_deps = list(dict.fromkeys(
                list(target_model.depends_on)
                + [check_name]
                + ([prev_retry_name] if prev_retry_name != target_name else [])
            ))

            extra_nodes.append(BaseModelHandler(
                name=retry_name,
                path=target_model.path,
                source=retry_source,
                depends_on=retry_deps,
                config=target_config,
                promptdata_used=target_model.promptdata_used,
                promptfiles_used=target_model.promptfiles_used,
            ))

            prev_retry_name = retry_name

        # --- updated self: terminal pass-through (keeps original name) ---
        terminal_source = f"{{{{ skip_and_set_to_value(ref('{prev_retry_name}')) }}}}"
        updated_self = BaseModelHandler(
            name=self.name,
            path=self.path,
            source=terminal_source,
            depends_on=[prev_retry_name],
            config={},
            promptdata_used=[],
            promptfiles_used=[],
        )

        return updated_self, extra_nodes
