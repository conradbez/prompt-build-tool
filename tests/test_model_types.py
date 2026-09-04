"""Model-type registry, the shared execution lifecycle, and the built-in types."""

from __future__ import annotations

import asyncio
import json
import warnings

import pytest

import pbt
from pbt.executor.executor import execute_run
from pbt.executor.graph import build_models_from_dict
from pbt.model_spec import ModelSpec
from pbt.model_types import get_model_type, known_model_types
from pbt.storage.memory import MemoryStorageBackend


def stub_llm(prompt: str, config: dict | None = None) -> str:
    if (config or {}).get("output_format") == "json":
        return json.dumps(["a", "b"])
    return "resp"


def run_models(models: dict[str, str], *, storage=None, llm_call=stub_llm, **kwargs):
    """Execute *models* and return (storage, run_id, {model_name: ModelRunResult})."""
    storage = storage or MemoryStorageBackend()
    storage.init_db()
    specs = list(build_models_from_dict(models).values())
    run_id = storage.create_run(model_count=len(specs))
    results = asyncio.run(execute_run(
        run_id=run_id,
        ordered_models=specs,
        storage_backend=storage,
        llm_call=llm_call,
        **kwargs,
    ))
    return storage, run_id, {r.model_name: r for r in results}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_builtin_types_are_registered():
    assert known_model_types() == {
        "template", "loop", "execute_python", "quality_check",
    }
    # The unnamed default is the plain LLM call.
    assert get_model_type("") is not None


def test_register_a_new_model_type_end_to_end():
    @pbt.model_type("shout_test", config_keys={"suffix"})
    class Shout(pbt.BaseModelType):
        async def execute(self, spec, ctx):
            rendered, state = ctx.render(spec)
            output = await ctx.call_llm(rendered, spec, state)
            return output.upper() + spec.config.get("suffix", "")

    _, _, results = run_models(
        {"s": '{{ config(model_type="shout_test", suffix="!") }}\nhello'}
    )
    assert results["s"].llm_output == "RESP!"


def test_registering_a_type_declares_its_config_keys():
    @pbt.model_type("quiet_test", config_keys={"volume"})
    class Quiet(pbt.BaseModelType):
        async def execute(self, spec, ctx):
            return "ok"

    assert "volume" in pbt.known_config_keys()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an UnknownConfigKeyWarning would fail here
        build_models_from_dict(
            {"q": '{{ config(model_type="quiet_test", volume="11") }}\nhi'}
        )


def test_unknown_model_type_warns_and_falls_back():
    with pytest.warns(pbt.UnknownConfigKeyWarning, match="unknown model_type 'nope'"):
        models = build_models_from_dict({"a": '{{ config(model_type="nope") }}\nHi'})
    assert models["a"].model_type == ""


def test_expansion_must_keep_the_declared_name():
    @pbt.model_type("bad_expand_test")
    class BadExpand(pbt.BaseModelType):
        def expand(self, spec, all_specs):
            return [spec.derive(name=f"{spec.name}_only", model_type="")]

    with pytest.raises(ValueError, match="without producing a node of that name"):
        build_models_from_dict({"x": '{{ config(model_type="bad_expand_test") }}\nHi'})


# ---------------------------------------------------------------------------
# Built-in types
# ---------------------------------------------------------------------------

def test_template_type_renders_without_calling_the_llm():
    calls: list[str] = []

    def counting_llm(prompt: str) -> str:
        calls.append(prompt)
        return "resp"

    _, _, results = run_models(
        {
            "src": "Name a topic.",
            "t": '{{ config(model_type="template") }}\nGot: {{ ref("src") }}',
        },
        llm_call=counting_llm,
    )
    assert "Got: resp" in results["t"].llm_output
    assert len(calls) == 1  # only 'src' reached the LLM


def test_quality_check_expands_and_keeps_the_declared_name():
    models = build_models_from_dict({
        "article": "Write an article.",
        "article_quality": (
            '{{ config(model_type="quality_check", quality_retries="2") }}\n'
            'Is {{ ref("article") }} good? Reply PASS or FAIL.'
        ),
    })
    assert set(models) == {
        "article",
        "article_quality_1", "article_1",
        "article_quality_2", "article_2",
        "article_quality",
    }
    # The node downstream models ref() is the terminal pass-through.
    assert models["article_quality"].depends_on == ["article_2"]


def test_loop_fans_out_over_a_json_list():
    _, _, results = run_models({
        "items": '{{ config(output_format="json") }}\nList things.',
        "each": '{{ config(model_type="loop") }}\nDescribe {{ ref("items") }}',
    })
    assert json.loads(results["each"].llm_output) == ["resp", "resp"]
    assert "[loop over 2 items from 'items']" in results["each"].prompt_rendered


# ---------------------------------------------------------------------------
# The shared lifecycle — behaviour every type gets for free
# ---------------------------------------------------------------------------

def test_execute_python_cache_hit_is_still_recorded_as_success():
    """A cached model must complete its run row, or `pbt test` cannot see it."""
    models = {"py": '{{ config(model_type="execute_python") }}\noutput = "hello"'}
    storage, _, _ = run_models(models)

    # Same storage, so the second run hits the prompt cache.
    _, run_id, results = run_models(models, storage=storage)

    assert results["py"].cached is True
    assert storage.get_run_results(run_id)[0]["status"] == "success"
    assert storage.get_model_outputs_from_run(run_id, ["py"]) == {"py": "hello"}


def test_execute_python_propagates_skip_this_and_downstream():
    _, _, results = run_models({
        "py": (
            '{{ config(model_type="execute_python") }}\n'
            '{{ skip_this_and_downstream("stop") }}'
        ),
        "after": 'Uses {{ ref("py") }}',
    })
    assert results["py"].prompt_skipped is True
    assert results["after"].status == "skipped"


def test_validated_output_is_what_downstream_readers_see():
    storage, run_id, _ = run_models(
        {"v": "say something"},
        llm_call=lambda prompt: "raw-output",
        validators={"v": lambda prompt, result: "VALIDATED"},
    )
    # `pbt test` reads outputs back out of storage — it must judge the value the
    # pipeline actually passed on, not the pre-validation text.
    assert storage.get_model_outputs_from_run(run_id, ["v"]) == {"v": "VALIDATED"}
    # ...while the prompt cache still holds the raw output, so editing a
    # validator does not force a fresh LLM call.
    assert list(storage._cache.values()) == ["raw-output"]


def test_cache_key_is_one_formula_for_every_type():
    from pbt.executor.run_context import RunContext

    ctx = RunContext(run_id="r", storage=MemoryStorageBackend(), llm_call=stub_llm)
    spec = ModelSpec(name="s", source="hi", config={"a": "1"})
    assert ctx.cache_key(spec, "rendered", None) == 'rendered\x00{"a": "1"}\x00'


def test_post_processing_type_is_idempotent_across_cached_runs():
    """A strategy that transforms the LLM response must not re-transform a cache hit.

    The cache has to hold the raw response, not this model's final output, or
    the second run applies the transform to an already-transformed value.
    """
    @pbt.model_type("suffixer_test", config_keys={"suffix"})
    class Suffixer(pbt.BaseModelType):
        async def execute(self, spec, ctx):
            rendered, state = ctx.render(spec)
            output = await ctx.call_llm(rendered, spec, state)
            return output + spec.config.get("suffix", "")

    models = {"s": '{{ config(model_type="suffixer_test", suffix="!") }}\nHeadline?'}

    storage, _, first = run_models(models)
    second_storage, run_id, second = run_models(models, storage=storage)

    assert first["s"].llm_output == "resp!"
    assert second["s"].cached is True
    assert second["s"].llm_output == "resp!"          # not "resp!!"
    assert list(storage._cache.values()) == ["resp"]  # cache holds the raw response
    # Readers of the run see the model's real output, not the raw response.
    assert second_storage.get_model_outputs_from_run(run_id, ["s"]) == {"s": "resp!"}
