"""Token usage: llm_call returning pbt.LLMResult, storage, and the docs report."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

import pbt
from pbt.executor.executor import execute_run
from pbt.executor.graph import build_models_from_dict
from pbt.storage.memory import MemoryStorageBackend
from pbt.storage.sqlite import SQLiteStorageBackend

MODELS = {
    "topic": "Name a topic.",
    "article": "Write about {{ ref('topic') }}.",
    "header": '{{ config(model_type="template") }}# {{ ref("topic") }}',
}


def token_llm(prompt: str) -> pbt.LLMResult:
    return pbt.LLMResult(f"re: {prompt}", input_tokens=10, output_tokens=5)


def run_once(storage, llm_call=token_llm, models=MODELS):
    specs = list(build_models_from_dict(models).values())
    run_id = storage.create_run(model_count=len(specs))
    results = asyncio.run(execute_run(
        run_id=run_id, ordered_models=specs, storage_backend=storage, llm_call=llm_call,
    ))
    storage.finish_run(run_id, "success")
    return run_id, {r.model_name: r for r in results}


@pytest.fixture(params=["memory", "sqlite"])
def storage(request, tmp_path):
    backend = MemoryStorageBackend() if request.param == "memory" else SQLiteStorageBackend(tmp_path / "pbt.db")
    backend.init_db()
    return backend


def test_llm_result_is_unwrapped_and_tokens_recorded(storage):
    run_id, results = run_once(storage)
    assert results["topic"].value == "re: Name a topic."
    assert results["article"].value == "re: Write about re: Name a topic.."
    assert (results["topic"].input_tokens, results["topic"].output_tokens) == (10, 5)
    assert results["topic"].cache_input_tokens is None
    # A template makes no call, so its tokens stay unknown.
    assert results["header"].input_tokens is None

    rows = {r["model_name"]: r for r in storage.get_run_results(run_id)}
    assert rows["topic"]["llm_output"] == "re: Name a topic."
    assert (rows["article"]["input_tokens"], rows["article"]["output_tokens"]) == (10, 5)
    assert rows["header"]["input_tokens"] is None


def test_cache_hit_reports_the_tokens_it_saved(storage):
    run_once(storage)
    run_id, results = run_once(storage)
    topic = results["topic"]
    assert topic.cached
    assert (topic.input_tokens, topic.output_tokens) == (None, None)
    assert (topic.cache_input_tokens, topic.cache_output_tokens) == (10, 5)

    # A third run's hit is served from the second run's (cached) row — the
    # original call's tokens carry forward.
    _, results = run_once(storage)
    assert (results["article"].cache_input_tokens, results["article"].cache_output_tokens) == (10, 5)


def test_plain_string_llm_call_leaves_tokens_unknown(storage):
    run_id, results = run_once(storage, llm_call=lambda p: "plain")
    assert results["topic"].value == "plain"
    assert results["topic"].input_tokens is None
    assert all(r["input_tokens"] is None for r in storage.get_run_results(run_id))


def test_json_output_parses_inside_llm_result(storage):
    models = {"items": '{{ config(output_format="json") }}List.'}
    _, results = run_once(
        storage, llm_call=lambda p: pbt.LLMResult('["a"]', input_tokens=3), models=models,
    )
    assert results["items"].value == ["a"]
    assert (results["items"].input_tokens, results["items"].output_tokens) == (3, None)


def test_older_database_gains_token_columns(tmp_path):
    path = tmp_path / "pbt.db"
    SQLiteStorageBackend(path).init_db()
    with sqlite3.connect(path) as conn:
        for col in ("input_tokens", "output_tokens", "cache_input_tokens", "cache_output_tokens"):
            conn.execute(f"ALTER TABLE model_results DROP COLUMN {col}")

    backend = SQLiteStorageBackend(path)
    backend.init_db()
    run_id, _ = run_once(backend)
    rows = {r["model_name"]: r for r in backend.get_run_results(run_id)}
    assert rows["topic"]["input_tokens"] == 10


def _docs(storage, tmp_path) -> str:
    from pbt.docs import generate_docs

    runs = storage.get_latest_runs()
    out = tmp_path / "index.html"
    generate_docs(
        runs=runs,
        run_results={r["run_id"]: storage.get_run_results(r["run_id"]) for r in runs},
        models=None,
        output_path=out,
    )
    return out.read_text()


def test_docs_show_tokens_used_cached_and_cold_start(tmp_path):
    storage = MemoryStorageBackend()
    run_once(storage)                                  # 2 calls x 15 tokens
    run_once(storage, models={**MODELS, "extra": "More."})  # 30 cached + 15 new
    html = _docs(storage, tmp_path)

    assert "Tokens used" in html and "From cache" in html and "Est. cold start" in html
    assert '<td class="num">15</td>' in html   # second run: used
    assert '<td class="num">45</td>' in html   # second run: cold start
    assert '<td class="num">30</td>' in html   # first run: used / cold; second: cached
    assert "Token usage is not being reported" not in html


def test_docs_explain_how_to_report_tokens_when_none_are(tmp_path):
    storage = MemoryStorageBackend()
    run_once(storage, llm_call=lambda p: "plain")
    html = _docs(storage, tmp_path)
    assert "Token usage is not being reported" in html
    assert "pbt.LLMResult" in html and "client.py" in html


def test_llm_judged_test_unwraps_llm_result(tmp_path):
    from pbt.tester import _invoke_llm

    verdict = _invoke_llm("judge", lambda p: pbt.LLMResult('{"results": "pass"}', input_tokens=1))
    assert verdict == '{"results": "pass"}'
