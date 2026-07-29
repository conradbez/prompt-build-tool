"""
config() key validation — unrecognised keys warn instead of being swallowed.

Also covers loop_over, which pins the dependency a loop model fans out over.
"""

from __future__ import annotations

import json

import pytest

import pbt
from pbt.executor.graph import build_models_from_dict, load_models
from pbt.executor.parser_initial import _extra_config_keys
from pbt.storage import MemoryStorageBackend


# ---------------------------------------------------------------------------
# Unknown keys warn
# ---------------------------------------------------------------------------

def test_unknown_key_warns():
    with pytest.warns(pbt.UnknownConfigKeyWarning, match="'tags' is not recognised"):
        build_models_from_dict({"a": '{{ config(tags="article") }}\nHi'})


def test_known_keys_do_not_warn(recwarn):
    build_models_from_dict({
        "a": '{{ config(output_format="json", output_extension="html") }}\nHi',
        "b": '{{ config(model_type="loop", loop_over="a") }}\n{{ ref("a") }}',
    })
    assert [w for w in recwarn if issubclass(w.category, pbt.UnknownConfigKeyWarning)] == []


def test_unknown_key_is_still_captured_in_config():
    """The warning is advisory — the key stays in config for llm_call to consume."""
    with pytest.warns(pbt.UnknownConfigKeyWarning):
        models = build_models_from_dict({"a": '{{ config(temperature="0.7") }}\nHi'})
    assert models["a"].config["temperature"] == "0.7"


def test_near_miss_key_suggests_correction():
    with pytest.warns(pbt.UnknownConfigKeyWarning, match=r"Did you mean 'output_format'\?"):
        build_models_from_dict({"a": '{{ config(output_fmt="json") }}\nHi'})


def test_one_warning_per_unknown_key():
    with pytest.warns(pbt.UnknownConfigKeyWarning) as record:
        build_models_from_dict({"a": '{{ config(foo="1", bar="2", output_format="json") }}\nHi'})
    assert len(record) == 2


def test_warning_names_the_model_and_file(tmp_path, recwarn):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "article.prompt").write_text('{{ config(tags="x") }}\nHi', encoding="utf-8")

    with pytest.warns(pbt.UnknownConfigKeyWarning, match="article") as record:
        load_models(models_dir)
    assert "article.prompt" in str(record[0].message)


def test_register_config_keys_silences_warning(recwarn):
    """Custom llm_call keys can be registered as recognised."""
    original = set(_extra_config_keys)
    try:
        pbt.register_config_keys("temperature")
        assert "temperature" in pbt.known_config_keys()
        build_models_from_dict({"a": '{{ config(temperature="0.7") }}\nHi'})
        assert [w for w in recwarn if issubclass(w.category, pbt.UnknownConfigKeyWarning)] == []
    finally:
        _extra_config_keys.clear()
        _extra_config_keys.update(original)


# ---------------------------------------------------------------------------
# Unknown model_type warns rather than silently running a plain LLM call
# ---------------------------------------------------------------------------

def test_unknown_model_type_warns():
    with pytest.warns(pbt.UnknownConfigKeyWarning, match="unknown model_type 'lop'"):
        models = build_models_from_dict({"a": '{{ config(model_type="lop") }}\nHi'})
    assert models["a"].model_type == ""  # fell back to the plain handler


@pytest.mark.parametrize("model_type", ["loop", "execute_python", "quality_check"])
def test_known_model_types_do_not_warn(model_type, recwarn):
    build_models_from_dict({
        "src": '{{ config(output_format="json") }}\nList things.',
        "a": f'{{{{ config(model_type="{model_type}") }}}}\n{{{{ ref("src") }}}}',
    })
    assert [w for w in recwarn if issubclass(w.category, pbt.UnknownConfigKeyWarning)] == []


# ---------------------------------------------------------------------------
# loop_over
# ---------------------------------------------------------------------------

LOOP_MODELS = {
    "one": '{{ config(output_format="json") }}\nList A.',
    "two": '{{ config(output_format="json") }}\nList B.',
    "fan": (
        '{{ config(model_type="loop", loop_over="two") }}\n'
        'Describe {{ ref("one") }} and {{ ref("two") }}'
    ),
}


def _llm(prompt: str, config: dict | None = None) -> str:
    if (config or {}).get("output_format") == "json":
        return json.dumps(["x", "y", "z"])
    return "described"


async def _run(models: dict, select: list[str] | None = None) -> dict:
    return await pbt.async_run(
        models_from_dict=models,
        select=select,
        llm_call=_llm,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )


@pytest.mark.asyncio
async def test_loop_over_disambiguates_multiple_list_deps():
    """Without loop_over this DAG is ambiguous and raises."""
    outputs = await _run(LOOP_MODELS)
    assert not isinstance(outputs["fan"], pbt.ModelError), outputs["fan"]
    # One call per item in 'two' (3 items), not per item in 'one'.
    assert json.loads(outputs["fan"]) == ["described", "described", "described"]


@pytest.mark.asyncio
async def test_ambiguous_loop_without_loop_over_errors():
    models = {**LOOP_MODELS, "fan": '{{ config(model_type="loop") }}\n{{ ref("one") }}{{ ref("two") }}'}
    outputs = await _run(models)
    assert isinstance(outputs["fan"], pbt.ModelError)
    assert "loop_over" in str(outputs["fan"])


@pytest.mark.asyncio
async def test_loop_over_non_dependency_errors():
    models = {
        **LOOP_MODELS,
        "fan": '{{ config(model_type="loop", loop_over="nope") }}\n{{ ref("one") }}{{ ref("two") }}',
    }
    outputs = await _run(models)
    assert isinstance(outputs["fan"], pbt.ModelError)
    assert "not a dependency" in str(outputs["fan"])


@pytest.mark.asyncio
async def test_loop_over_non_list_dependency_errors():
    models = {
        "text": "Just prose.",
        "fan": (
            '{{ config(model_type="loop", loop_over="text") }}\n'
            'Describe {{ ref("text") }}'
        ),
    }
    outputs = await _run(models)
    assert isinstance(outputs["fan"], pbt.ModelError)
    assert "does not return a JSON list" in str(outputs["fan"])
