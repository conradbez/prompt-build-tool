"""
``config(each='path')`` fan-out and path ``ref('model.key[0]')``.

Map is ``each=`` on any model kind, map over map is ``[*][*]`` in the path,
reduce is any model that ``ref()``s the mapped list.  Each item is stored as
its own ``model[i]`` row, which is also its prompt-cache entry.
"""

from __future__ import annotations

import json

import pytest

from pbt import jsonpath
from pbt.executor.parser_initial import extract_dependencies
from pbt.storage import MemoryStorageBackend
from tests.test_loop import run_models


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def test_parse_splits_model_and_steps():
    assert jsonpath.parse("parts") == ("parts", [])
    assert jsonpath.parse("parts.items[0].name") == ("parts", ["items", 0, "name"])
    assert jsonpath.parse("sections[*][*]") == ("sections", ["*", "*"])


@pytest.mark.parametrize("bad", ["", "a..b", "a[x]", "a.b-c", "a[*"])
def test_parse_rejects_bad_paths(bad):
    with pytest.raises(ValueError, match="not a valid path"):
        jsonpath.parse(bad)


def test_wildcards_unnest_one_level_each():
    sections = [["s1", "s2"], ["s3"]]
    assert jsonpath.resolve(sections, ["*"], "x") == sections
    assert jsonpath.resolve(sections, ["*", "*"], "x") == ["s1", "s2", "s3"]
    parts = {"parts": [{"id": 1}, {"id": 2}]}
    assert jsonpath.resolve(parts, ["parts", "*", "id"], "x") == [1, 2]
    assert jsonpath.resolve(parts, ["parts", "1", "id"], "x") == 2


def test_resolve_is_strict_and_names_what_it_found():
    with pytest.raises(ValueError, match=r"\.part not found in a dict with keys \['parts'\]"):
        jsonpath.resolve({"parts": []}, ["part"], "ref('p.part')")
    with pytest.raises(ValueError, match=r"\[\*\] expects a list, got a dict"):
        jsonpath.resolve({"a": 1}, ["*"], "x")


def test_dependency_is_the_leading_name_of_a_path():
    src = "{{ ref('parts.parts[0].lcsc') }} {{ ref(\"brief\") }} {{ ref('sections[*][*]') }}"
    assert extract_dependencies(src) == ["parts", "brief", "sections"]


# ---------------------------------------------------------------------------
# A small map / map-over-map / reduce pipeline
# ---------------------------------------------------------------------------

PIPELINE = {
    "parts": '{{ config(output_format="json") }}\nPARTS',
    "chapters": '{{ config(output_format="json") }}\nCHAPTERS',
    # map, on a non-LLM kind, over a list nested under a key
    "check": (
        '{{ config(model_type="execute_python", each="parts.parts[*]") }}\n'
        "print('ok:' + {{ ref('parts').id | tojson }})"
    ),
    # map whose items are themselves lists
    "sections": '{{ config(each="chapters", output_format="json") }}\nSPLIT {{ ref("chapters") }}',
    # map over map: [*][*] flattens one level per wildcard
    "polish": '{{ config(each="sections[*][*]") }}\nPOLISH {{ ref("sections") }}',
    # reduce: a plain model sees the whole lists
    "report": "REPORT {{ ref('polish') | join(',') }} / {{ ref('check') | join(',') }}",
}


class CountingLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "PARTS" in prompt:
            return json.dumps({"parts": [{"id": "r1"}, {"id": "c1"}]})
        if "CHAPTERS" in prompt:
            return json.dumps(["ch1", "ch2"])
        if "SPLIT" in prompt:
            ch = prompt.split()[-1]
            return json.dumps([f"{ch}.a", f"{ch}.b"] if ch == "ch1" else [f"{ch}.a"])
        if "POLISH" in prompt:
            return "t:" + prompt.split()[-1]
        return prompt


def test_map_map_reduce_pipeline():
    llm = CountingLLM()
    _, _, results = run_models(PIPELINE, llm_call=llm)

    assert results["check"].value == ["ok:r1", "ok:c1"]
    assert results["sections"].value == [["ch1.a", "ch1.b"], ["ch2.a"]]
    assert results["polish"].value == ["t:ch1.a", "t:ch1.b", "t:ch2.a"]
    assert results["report"].value == "REPORT t:ch1.a,t:ch1.b,t:ch2.a / ok:r1,ok:c1"
    assert "[loop over 3 items from 'sections[*][*]']" in results["polish"].prompt_rendered


def test_each_adds_the_dependency_it_iterates():
    models = {  # dependent first: only the each= edge can order these
        "each": '{{ config(each="items") }}no ref here',
        "items": '{{ config(output_format="json") }}\nCHAPTERS',
    }
    _, _, results = run_models(models, llm_call=CountingLLM())
    assert results["each"].status == "success"
    assert results["each"].value == ["no ref here", "no ref here"]


def test_each_on_a_dict_errors_with_its_keys():
    models = {"parts": PIPELINE["parts"], "bad": '{{ config(each="parts") }}\nx'}
    _, _, results = run_models(models, llm_call=CountingLLM())
    assert results["bad"].status == "error"
    assert "dict with keys ['parts']" in results["bad"].error


def test_ref_path_inside_a_template():
    models = {
        "parts": PIPELINE["parts"],
        "first": '{{ config(model_type="template") }}{{ ref("parts.parts[0].id") }}',
    }
    _, _, results = run_models(models, llm_call=CountingLLM())
    assert results["first"].value == "r1"


# ---------------------------------------------------------------------------
# Item rows and per-item caching
# ---------------------------------------------------------------------------

def test_each_item_gets_its_own_row():
    storage, run_id, _ = run_models(PIPELINE, llm_call=CountingLLM())
    rows = {r["model_name"]: r for r in storage.get_run_results(run_id)}
    assert [n for n in rows if n.startswith("polish[")] == ["polish[0]", "polish[1]", "polish[2]"]
    assert "POLISH ch1.b" in rows["polish[1]"]["prompt_rendered"]
    assert rows["polish[1]"]["llm_output"] == "t:ch1.b"


def test_unchanged_items_are_served_from_cache():
    storage = MemoryStorageBackend()
    run_models(PIPELINE, storage=storage, llm_call=CountingLLM())

    second = CountingLLM()
    _, run_id, results = run_models(PIPELINE, storage=storage, llm_call=second)
    assert second.prompts == []
    assert results["polish"].cached
    assert results["polish"].value == ["t:ch1.a", "t:ch1.b", "t:ch2.a"]
    rows = {r["model_name"]: r for r in storage.get_run_results(run_id)}
    assert rows["polish[0]"]["cached"]
