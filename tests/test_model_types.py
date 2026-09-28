"""Model-kind registry, the shared execution lifecycle, and the built-in kinds."""

from __future__ import annotations

import asyncio
import json
import warnings

import pytest

import pbt
from pbt.executor.executor import execute_run
from pbt.executor.graph import build_models_from_dict
from pbt.model_spec import ModelSpec
from pbt.model_types import get_model_kind, known_model_kinds
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

def test_builtin_kinds_are_registered():
    assert known_model_kinds() == {"template", "execute_python", "agent", "quality"}
    # The unnamed default is the plain LLM call.
    assert get_model_kind("") is not None


def test_register_a_new_model_kind_end_to_end():
    @pbt.model_kind("shout_test", config_keys={"suffix"})
    async def shout(rendered, call):
        response = await call.llm(rendered)
        return response.upper() + call.spec.config.get("suffix", "")

    _, _, results = run_models(
        {"s": '{{ config(model_type="shout_test", suffix="!") }}\nhello'}
    )
    assert results["s"].llm_output == "RESP!"
    # Registration returns the function unchanged, so it stays callable alone.
    assert shout.__name__ == "shout"


def test_a_kind_with_no_exec_fn_uses_the_rendered_text_as_its_output():
    calls: list[str] = []
    pbt.register_model_kind(pbt.ModelKind("passthrough_test", exec_fn=None))

    _, _, results = run_models(
        {
            "src": "Name a topic.",
            "p": '{{ config(model_type="passthrough_test") }}\nGot: {{ ref("src") }}',
        },
        llm_call=lambda prompt: calls.append(prompt) or "resp",
    )
    assert results["p"].llm_output.strip() == "Got: resp"
    assert len(calls) == 1  # only 'src' reached the LLM


def test_registering_a_kind_declares_its_config_keys():
    @pbt.model_kind("quiet_test", config_keys={"volume"})
    async def quiet(rendered, call):
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


# ---------------------------------------------------------------------------
# Built-in types
# ---------------------------------------------------------------------------

def test_template_kind_renders_without_calling_the_llm():
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


# ---------------------------------------------------------------------------
# The shared lifecycle — behaviour every kind gets for free
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


def test_cache_key_is_one_formula_for_every_kind():
    from pbt.executor.run_context import RunContext

    ctx = RunContext(run_id="r", storage=MemoryStorageBackend(), llm_call=stub_llm)
    spec = ModelSpec(name="s", source="hi", config={"a": "1"})
    assert ctx.cache_key(spec, "rendered", None) == 'rendered\x00{"a": "1"}\x00'


def test_post_processing_kind_is_idempotent_across_cached_runs():
    """A kind that transforms the LLM response must not re-transform a cache hit.

    The cache has to hold the raw response, not this model's final output, or
    the second run applies the transform to an already-transformed value.
    """
    @pbt.model_kind("suffixer_test", config_keys={"suffix"})
    async def suffixer(rendered, call):
        response = await call.llm(rendered)
        return response + call.spec.config.get("suffix", "")

    models = {"s": '{{ config(model_type="suffixer_test", suffix="!") }}\nHeadline?'}

    storage, _, first = run_models(models)
    second_storage, run_id, second = run_models(models, storage=storage)

    assert first["s"].llm_output == "resp!"
    assert second["s"].cached is True
    assert second["s"].llm_output == "resp!"          # not "resp!!"
    assert list(storage._cache.values()) == ["resp"]  # cache holds the raw response
    # Readers of the run see the model's real output, not the raw response.
    assert second_storage.get_model_outputs_from_run(run_id, ["s"]) == {"s": "resp!"}


# ---------------------------------------------------------------------------
# loop_over — any kind, once per item
# ---------------------------------------------------------------------------

def test_loop_over_runs_the_llm_once_per_item():
    seen = []

    def llm(prompt: str, config: dict | None = None) -> str:
        if (config or {}).get("output_format") == "json":
            return json.dumps(["apple", "pear"])
        seen.append(prompt.strip())
        return f"about {prompt.split()[-1]}"

    _, _, results = run_models({
        "fruits": '{{ config(output_format="json") }}\nList fruit.',
        "notes": '{{ config(loop_over="fruits") }}\nDescribe {{ ref("fruits") }}',
        "digest": 'Notes: {{ ref("notes") }}',
    }, llm_call=llm)

    assert sorted(seen[:2]) == ["Describe apple", "Describe pear"]
    assert results["notes"].value == ["about apple", "about pear"]
    assert "Describe apple" in results["notes"].prompt_rendered
    assert "Describe pear" in results["notes"].prompt_rendered
    assert results["digest"].status == "success"


def test_loop_over_works_for_any_kind_and_item_fields():
    _, _, results = run_models({
        "people": (
            '{{ config(model_type="execute_python", output_format="json") }}\n'
            'import json\nprint(json.dumps([{"name": "ada"}, {"name": "alan"}]))'
        ),
        "greetings": (
            '{{ config(model_type="template", loop_over="people") }}'
            'Hi {{ ref("people").name }}'
        ),
    })
    assert results["greetings"].value == ["Hi ada", "Hi alan"]


def test_loop_over_parses_each_item_as_json():
    _, _, results = run_models({
        "xs": '{{ config(output_format="json") }}\nlist',
        "ys": '{{ config(loop_over="xs", output_format="json") }}\n{{ ref("xs") }}',
    })
    assert results["ys"].value == [["a", "b"], ["a", "b"]]


def test_loop_over_skip_replaces_only_that_item():
    _, _, results = run_models({
        "xs": '{{ config(output_format="json") }}\nlist',
        "ys": (
            '{{ config(loop_over="xs") }}\n'
            '{% if ref("xs") == "a" %}{{ skip_and_set_to_value("skipped") }}{% endif %}'
            '{{ ref("xs") }}'
        ),
    })
    assert results["ys"].value == ["skipped", "resp"]
    assert results["ys"].prompt_skipped is False


def test_loop_over_needs_a_list():
    _, _, results = run_models({
        "x": "text",
        "y": '{{ config(loop_over="x") }}\n{{ ref("x") }}',
    })
    assert results["y"].status == "error"
    assert "needs a list" in results["y"].error


def test_loop_over_must_be_a_dependency():
    _, _, results = run_models({
        "xs": '{{ config(output_format="json") }}\nlist',
        "y": '{{ config(loop_over="xs") }}\nno ref here',
    })
    assert results["y"].status == "error"
    assert "not a dependency" in results["y"].error


# ---------------------------------------------------------------------------
# quality — the LLM call, checked and retried
# ---------------------------------------------------------------------------

def _quality_llm(verdicts: list[str]):
    """An LLM that answers checks from *verdicts* in turn, and numbers attempts."""
    prompts: list[str] = []

    def llm(prompt: str, config: dict | None = None) -> str:
        prompts.append(prompt.strip())
        if prompt.startswith("Check this answer"):
            return verdicts.pop(0)
        return f"attempt {sum(not p.startswith('Check') for p in prompts)}"

    return llm, prompts


def test_quality_returns_first_answer_that_passes():
    llm, prompts = _quality_llm(["FAIL: too short", "PASS"])
    _, _, results = run_models({
        "essay": '{{ config(model_type="quality", quality_check="Long enough") }}\nWrite.',
    }, llm_call=llm)

    assert results["essay"].value == "attempt 2"
    assert len(prompts) == 4  # attempt, check, retry, check
    assert "Long enough" in prompts[1]
    assert "FAIL: too short" in prompts[2] and prompts[2].startswith("Write.")


def test_quality_stops_after_retries_and_keeps_the_last_attempt():
    llm, prompts = _quality_llm(["no", "no", "no"])
    _, _, results = run_models({
        "essay": (
            '{{ config(model_type="quality", quality_check="c", quality_retries="1") }}\n'
            'Write.'
        ),
    }, llm_call=llm)
    assert results["essay"].value == "attempt 2"
    assert len(prompts) == 3  # attempt, check, retry


def test_quality_needs_a_check():
    _, _, results = run_models({"essay": '{{ config(model_type="quality") }}\nWrite.'})
    assert results["essay"].status == "error"
    assert "quality_check" in results["essay"].error
