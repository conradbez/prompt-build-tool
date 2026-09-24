"""
config() key validation — unrecognised keys warn instead of being swallowed.
"""

from __future__ import annotations

import pytest

import pbt
from pbt.executor.graph import build_models_from_dict, load_models
from pbt.executor.parser_initial import _extra_config_keys


# ---------------------------------------------------------------------------
# Unknown keys warn
# ---------------------------------------------------------------------------

def test_unknown_key_warns():
    with pytest.warns(pbt.UnknownConfigKeyWarning, match="'tags' is not recognised"):
        build_models_from_dict({"a": '{{ config(tags="article") }}\nHi'})


def test_known_keys_do_not_warn(recwarn):
    build_models_from_dict({
        "a": '{{ config(output_format="json", output_extension="html") }}\nHi',
        "b": '{{ config(model_type="template", global_instruction=False) }}\n{{ ref("a") }}',
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


@pytest.mark.parametrize("model_type", ["template", "execute_python"])
def test_known_model_types_do_not_warn(model_type, recwarn):
    build_models_from_dict({
        "src": '{{ config(output_format="json") }}\nList things.',
        "a": f'{{{{ config(model_type="{model_type}") }}}}\n{{{{ ref("src") }}}}',
    })
    assert [w for w in recwarn if issubclass(w.category, pbt.UnknownConfigKeyWarning)] == []
