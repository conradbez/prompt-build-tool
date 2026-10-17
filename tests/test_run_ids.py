"""Content hashes on runs (template, map, run) and ``{{ previous }}`` in templates."""

from __future__ import annotations

import io
import json

import pytest

import promptmap
from promptmap.executor.graph import build_models_from_dict
from promptmap.run_ids import map_hash, run_hash, template_hash
from promptmap.storage import MemoryStorageBackend
from promptmap.storage.sqlite import SQLiteStorageBackend


class EchoLLM:
    """Answers with a counter, and remembers every prompt it was sent."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return f"answer {len(self.prompts)}"


@pytest.fixture(params=["memory", "sqlite"])
def storage(request, tmp_path):
    if request.param == "memory":
        return MemoryStorageBackend()
    return SQLiteStorageBackend(tmp_path / "promptmap.db")


def _run(models: dict[str, str], storage, llm, **kwargs) -> dict:
    return promptmap.run(
        models_from_dict=models, storage_backend=storage, llm_call=llm, verbose=False, **kwargs
    )


def _latest_run(storage):
    return storage.get_latest_runs(1)[0]


def _row(storage, run_id: str, name: str):
    return next(r for r in storage.get_run_results(run_id) if r["model_name"] == name)


# -- the hashes -------------------------------------------------------------


def test_every_result_row_stores_its_template_hash(storage) -> None:
    source = "Say hi to {{ promptdata('who') }}"
    _run({"a": source}, storage, EchoLLM(), promptdata={"who": "Ada"})

    run = _latest_run(storage)
    assert _row(storage, run["run_id"], "a")["template_hash"] == template_hash(source)


def test_run_ids_are_stored_and_stable_across_identical_runs(storage) -> None:
    models = {"a": "Hi", "b": "{{ ref('a') }} again"}
    _run(models, storage, EchoLLM(), promptdata={"x": "1"})
    first = _latest_run(storage)
    _run(models, storage, EchoLLM(), promptdata={"x": "1"})
    second = _latest_run(storage)

    assert first["run_id"] != second["run_id"]
    assert first["map_hash"] and first["map_hash"] == second["map_hash"]
    assert first["run_hash"] and first["run_hash"] == second["run_hash"]


def test_promptdata_changes_the_run_hash_but_not_the_map_hash(storage) -> None:
    models = {"a": "Hi {{ promptdata('x') }}"}
    _run(models, storage, EchoLLM(), promptdata={"x": "1"})
    first = _latest_run(storage)
    _run(models, storage, EchoLLM(), promptdata={"x": "2"})
    second = _latest_run(storage)

    assert first["map_hash"] == second["map_hash"]
    assert first["run_hash"] != second["run_hash"]


def test_editing_any_prompt_changes_both_run_level_hashes(storage) -> None:
    _run({"a": "Hi", "b": "Bye"}, storage, EchoLLM())
    first = _latest_run(storage)
    _run({"a": "Hi", "b": "Bye!"}, storage, EchoLLM())
    second = _latest_run(storage)

    assert first["map_hash"] != second["map_hash"]
    assert first["run_hash"] != second["run_hash"]


def test_map_hash_covers_test_files(tmp_path) -> None:
    models = build_models_from_dict({"a": "Hi"}).values()
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "a_is_short.prompt").write_text("Is {{ ref('a') }} short?")
    before = map_hash(models, tests)

    (tests / "a_is_short.prompt").write_text("Is {{ ref('a') }} very short?")

    assert map_hash(models, tests) != before
    assert map_hash(models, tests) != map_hash(models, None)


def test_run_hash_covers_promptfile_contents() -> None:
    models = build_models_from_dict({"a": "Hi"}).values()
    one = run_hash("m", models, promptfiles={"doc": io.BytesIO(b"one")})
    two = run_hash("m", models, promptfiles={"doc": io.BytesIO(b"two")})
    again = run_hash("m", models, promptfiles={"doc": io.BytesIO(b"one")})

    assert one != two
    assert one == again


def test_hashing_a_promptfile_leaves_it_readable() -> None:
    handle = io.BytesIO(b"contents")
    run_hash("m", [], promptfiles={"doc": handle})
    assert handle.read() == b"contents"


# -- {{ previous }} ---------------------------------------------------------


_ITERATE = (
    "{% if previous %}Improve on: {{ previous.output }} (asked: {{ previous.prompt }})"
    "{% else %}First draft{% endif %}"
)


def test_previous_is_none_on_the_first_run(storage) -> None:
    llm = EchoLLM()
    _run({"a": _ITERATE}, storage, llm)
    assert llm.prompts == ["First draft"]


def test_previous_holds_the_last_rendered_prompt_and_output(storage) -> None:
    llm = EchoLLM()
    _run({"a": _ITERATE}, storage, llm)
    _run({"a": _ITERATE}, storage, llm)

    assert llm.prompts[1] == "Improve on: answer 1 (asked: First draft)"


def test_previous_follows_the_latest_run(storage) -> None:
    llm = EchoLLM()
    for _ in range(3):
        _run({"a": _ITERATE}, storage, llm)

    assert llm.prompts[2].startswith("Improve on: answer 2 (asked: Improve on: answer 1")


def test_previous_is_hidden_after_the_template_changes(storage) -> None:
    llm = EchoLLM()
    _run({"a": _ITERATE}, storage, llm)
    _run({"a": _ITERATE + " v2"}, storage, llm)

    assert llm.prompts[1] == "First draft v2"


def test_previous_comes_back_for_a_template_that_returns(storage) -> None:
    """The lookup is by hash, not by most recent run of the model."""
    llm = EchoLLM()
    _run({"a": _ITERATE}, storage, llm)
    _run({"a": _ITERATE + " v2"}, storage, llm)
    _run({"a": _ITERATE}, storage, llm)

    assert llm.prompts[2] == "Improve on: answer 1 (asked: First draft)"


def test_previous_is_per_model(storage) -> None:
    llm = EchoLLM()
    _run({"a": _ITERATE}, storage, llm)
    _run({"b": _ITERATE}, storage, llm)

    run = _latest_run(storage)
    assert _row(storage, run["run_id"], "b")["prompt_rendered"] == "First draft"


def test_previous_output_is_parsed_for_json_models(storage) -> None:
    source = (
        "{{ config(output_format='json') }}"
        "{% if previous %}last count {{ previous.output.count }}{% else %}start{% endif %}"
    )
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"count": len(prompts)})

    _run({"a": source}, storage, llm)
    _run({"a": source}, storage, llm)

    assert prompts == ["start", "last count 1"]


def test_previous_ignores_failed_runs(storage) -> None:
    def failing(prompt: str) -> str:
        raise RuntimeError("down")

    llm = EchoLLM()
    _run({"a": _ITERATE}, storage, llm)
    _run({"a": _ITERATE}, storage, failing)
    _run({"a": _ITERATE}, storage, llm)

    assert llm.prompts[1] == "Improve on: answer 1 (asked: First draft)"
