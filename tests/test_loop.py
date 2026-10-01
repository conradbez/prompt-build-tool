"""
The ``loop`` model kind and the ``fan_out`` hook behind it.

A loop model renders once per item of an upstream JSON list (or the files of a
Dir), with ``ref('<list model>')`` yielding the current item, and collects the
results into a list in input order.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import pbt
from pbt.executor.executor import execute_run
from pbt.executor.graph import build_models_from_dict
from pbt.files import Dir, File
from pbt.model_types import _REGISTRY
from pbt.storage import MemoryStorageBackend


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
# Fan-out
# ---------------------------------------------------------------------------

def test_loop_fans_out_over_a_json_list():
    _, _, results = run_models({
        "items": '{{ config(output_format="json") }}\nList things.',
        "each": '{{ config(model_type="loop") }}\nDescribe {{ ref("items") }}',
    })
    assert json.loads(results["each"].llm_output) == ["resp", "resp"]
    assert "[loop over 2 items from 'items']" in results["each"].prompt_rendered


def test_ref_yields_the_current_item_in_order():
    prompts: list[str] = []

    def llm(prompt: str, config: dict | None = None) -> str:
        prompts.append(prompt)
        if (config or {}).get("output_format") == "json":
            return json.dumps(["apple", "banana", "cherry"])
        return prompt.split()[-1].upper()

    _, _, results = run_models(
        {
            "items": '{{ config(output_format="json") }}\nList fruit.',
            "each": '{{ config(model_type="loop") }}\nShout {{ ref("items") }}',
        },
        llm_call=llm,
    )
    assert results["each"].value == ["APPLE", "BANANA", "CHERRY"]
    assert sorted(p.strip() for p in prompts if "Shout" in p) == [
        "Shout apple", "Shout banana", "Shout cherry",
    ]


def test_one_skipped_item_does_not_skip_the_loop():
    _, _, results = run_models({
        "items": '{{ config(output_format="json") }}\nList things.',
        "each": (
            '{{ config(model_type="loop") }}\n'
            '{% if ref("items") == "a" %}{{ skip_and_set_to_value("skipped") }}{% endif %}'
            'Describe {{ ref("items") }}'
        ),
    })
    assert results["each"].value == ["skipped", "resp"]
    assert results["each"].status != pbt.ModelStatus.SKIPPED


def test_loop_json_output_parses_each_item():
    def llm(prompt: str, config: dict | None = None) -> str:
        if "List" in prompt:
            return json.dumps(["a", "b"])
        return json.dumps({"item": prompt.split()[-1]})

    _, _, results = run_models(
        {
            "items": '{{ config(output_format="json") }}\nList things.',
            "each": (
                '{{ config(model_type="loop", output_format="json") }}\n'
                'Wrap {{ ref("items") }}'
            ),
        },
        llm_call=llm,
    )
    assert results["each"].value == [{"item": "a"}, {"item": "b"}]


def test_custom_kind_can_fan_out():
    async def shout(rendered, call):
        return (await call.llm(rendered)).upper()

    pbt.register_model_kind(pbt.ModelKind("shout_each_test", exec_fn=shout, fan_out=True))
    try:
        _, _, results = run_models({
            "items": '{{ config(output_format="json") }}\nList things.',
            "each": '{{ config(model_type="shout_each_test") }}\n{{ ref("items") }}',
        })
    finally:
        _REGISTRY.pop("shout_each_test", None)
    assert results["each"].value == ["RESP", "RESP"]


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def test_loop_attaches_each_items_file():
    def llm(prompt, files=None, config=None):
        if prompt.startswith("draw"):
            return [File(b"one", name="1.png"), File(b"two", name="2.png")]
        return ",".join(f.name for f in files or [])

    _, _, results = run_models(
        {
            "imgs": "draw two",
            "each": (
                '{{ config(model_type="loop", promptfiles=["imgs"]) }}\n'
                "Critique {{ ref('imgs') }}"
            ),
        },
        llm_call=llm,
    )
    assert results["each"].value == ["1.png", "2.png"]


def test_loop_over_a_dir():
    def llm(prompt, files=None, config=None):
        if prompt.startswith("draw"):
            return Dir({"a.txt": b"a", "b.txt": b"b"}, name="d")
        return prompt.split()[-1]

    _, _, results = run_models(
        {"d": "draw", "each": '{{ config(model_type="loop") }}\nname {{ ref("d").name }}'},
        llm_call=llm,
    )
    assert results["each"].value == ["a.txt", "b.txt"]


# ---------------------------------------------------------------------------
# Global instruction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_model_gets_global_instruction_per_item():
    prompts: list[str] = []

    def llm(prompt: str, config: dict | None = None) -> str:
        prompts.append(prompt)
        return stub_llm(prompt, config)

    await pbt.async_run(
        models_from_dict={
            "items": '{{ config(output_format="json") }}\nList things.',
            "items_loop": '{{ config(model_type="loop") }}\nDescribe: {{ ref("items") }}',
        },
        llm_call=llm,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
        global_instruction="GLOBAL.",
    )
    loop_prompts = [p for p in prompts if p.startswith("GLOBAL.\n\n") and "Describe:" in p]
    assert len(loop_prompts) == 2, prompts


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


async def _run(models: dict) -> dict:
    return await pbt.async_run(
        models_from_dict=models,
        llm_call=_llm,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )


@pytest.mark.asyncio
async def test_loop_over_disambiguates_multiple_list_deps():
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
        "fan": '{{ config(model_type="loop", loop_over="text") }}\nDescribe {{ ref("text") }}',
    }
    outputs = await _run(models)
    assert isinstance(outputs["fan"], pbt.ModelError)
    assert "does not return a JSON list" in str(outputs["fan"])
