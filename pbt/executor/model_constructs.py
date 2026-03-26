"""
Concrete model-type handlers for ``normal``, ``loop``, and ``execute_python``.

Each class inherits from ``BaseModelHandler`` and overrides ``execute_node``
with the appropriate execution strategy.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import re
import time
from contextlib import redirect_stdout
from typing import TYPE_CHECKING, Callable

from pbt.executor.model_type_registry import BaseModelHandler
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


class NormalModelHandler(BaseModelHandler):
    """Handles plain LLM calls (the default model type)."""

    model_type: str = ""  # type: ignore[assignment]

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

        rendered, skip_state = render_prompt(self.source, model_outputs, promptdata=promptdata, rag_call=rag_call, prompt_skipped_models=prompt_skipped_models)
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

            if asyncio.iscoroutinefunction(llm_call):
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
