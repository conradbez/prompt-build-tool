"""
Tests for per-test config() parameterisation.

Covers:
- extract_test_config: parses promptdata and promptfiles from config() calls
- get_test_row_count: validates consistent row counts across tests
- get_merged_row_params: merges promptdata/promptfiles for a given row
- execute_tests: per-test config drives promptdata injection and row naming
- CLI pbt test: config rows trigger model re-runs per row
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pbt.executor.parser_initial import extract_test_config
from pbt.tester import execute_tests, get_test_row_count, get_merged_row_params, load_tests
from pbt.storage.memory import MemoryStorageBackend
from tests.conftest import run_pbt


# ---------------------------------------------------------------------------
# extract_test_config
# ---------------------------------------------------------------------------

def test_extract_test_config_empty() -> None:
    assert extract_test_config("No config here.") == {}


def test_extract_test_config_single_promptdata() -> None:
    src = '{{ config(promptdata={"topic": "AI"}) }}\nSome test.'
    cfg = extract_test_config(src)
    assert cfg["promptdata"] == {"topic": "AI"}


def test_extract_test_config_single_promptfiles() -> None:
    src = '{{ config(promptfiles={"doc": "report.pdf"}) }}\nSome test.'
    cfg = extract_test_config(src)
    assert cfg["promptfiles"] == {"doc": "report.pdf"}


def test_extract_test_config_list_promptdata() -> None:
    src = '{{ config(promptdata=[{"topic": "AI"}, {"topic": "ML"}]) }}\nSome test.'
    cfg = extract_test_config(src)
    assert cfg["promptdata"] == [{"topic": "AI"}, {"topic": "ML"}]


def test_extract_test_config_list_promptfiles() -> None:
    src = '{{ config(promptfiles=[{"doc": "a.pdf"}, {"doc": "b.pdf"}]) }}\nSome test.'
    cfg = extract_test_config(src)
    assert cfg["promptfiles"] == [{"doc": "a.pdf"}, {"doc": "b.pdf"}]


def test_extract_test_config_both_list() -> None:
    src = (
        '{{ config('
        'promptdata=[{"topic": "AI"}, {"topic": "ML"}],'
        'promptfiles=[{"doc": "a.pdf"}, {"doc": "b.pdf"}]'
        ') }}\nSome test.'
    )
    cfg = extract_test_config(src)
    assert cfg["promptdata"] == [{"topic": "AI"}, {"topic": "ML"}]
    assert cfg["promptfiles"] == [{"doc": "a.pdf"}, {"doc": "b.pdf"}]


# ---------------------------------------------------------------------------
# get_test_row_count
# ---------------------------------------------------------------------------

def test_get_test_row_count_no_config() -> None:
    tests = {"t": "No config here."}
    assert get_test_row_count(tests) == 0


def test_get_test_row_count_single_dict() -> None:
    tests = {"t": '{{ config(promptdata={"k": "v"}) }}\ntest'}
    assert get_test_row_count(tests) == 0


def test_get_test_row_count_list() -> None:
    tests = {"t": '{{ config(promptdata=[{"k": "a"}, {"k": "b"}]) }}\ntest'}
    assert get_test_row_count(tests) == 2


def test_get_test_row_count_consistent() -> None:
    src2 = '{{ config(promptdata=[{"k": "a"}, {"k": "b"}]) }}\ntest'
    tests = {"a": src2, "b": src2}
    assert get_test_row_count(tests) == 2


def test_get_test_row_count_inconsistent_raises() -> None:
    src2 = '{{ config(promptdata=[{"k": "a"}, {"k": "b"}]) }}\ntest'
    src3 = '{{ config(promptdata=[{"k": "a"}, {"k": "b"}, {"k": "c"}]) }}\ntest'
    tests = {"a": src2, "b": src3}
    with pytest.raises(ValueError, match="same number of rows"):
        get_test_row_count(tests)


def test_get_test_row_count_mismatched_pd_pf_raises() -> None:
    src = (
        '{{ config('
        'promptdata=[{"k": "a"}, {"k": "b"}],'
        'promptfiles=[{"f": "x.pdf"}]'
        ') }}\ntest'
    )
    tests = {"t": src}
    with pytest.raises(ValueError, match="same length"):
        get_test_row_count(tests)


# ---------------------------------------------------------------------------
# get_merged_row_params
# ---------------------------------------------------------------------------

def test_get_merged_row_params_combines_tests() -> None:
    src_a = '{{ config(promptdata=[{"topic": "AI"}, {"topic": "ML"}]) }}\ntest'
    src_b = '{{ config(promptdata=[{"tone": "formal"}, {"tone": "casual"}]) }}\ntest'
    tests = {"a": src_a, "b": src_b}
    pd0, pf0 = get_merged_row_params(tests, 0)
    assert pd0 == {"topic": "AI", "tone": "formal"}
    pd1, pf1 = get_merged_row_params(tests, 1)
    assert pd1 == {"topic": "ML", "tone": "casual"}


def test_get_merged_row_params_single_dict_included_in_all_rows() -> None:
    src_list = '{{ config(promptdata=[{"topic": "AI"}, {"topic": "ML"}]) }}\ntest'
    src_dict = '{{ config(promptdata={"lang": "en"}) }}\ntest'
    tests = {"a": src_list, "b": src_dict}
    pd0, _ = get_merged_row_params(tests, 0)
    assert pd0["lang"] == "en"
    pd1, _ = get_merged_row_params(tests, 1)
    assert pd1["lang"] == "en"


# ---------------------------------------------------------------------------
# execute_tests with row_index
# ---------------------------------------------------------------------------

PASS_LLM = lambda prompt: json.dumps({"results": "pass"})

SIMPLE_TEST = '{{ config(promptdata=[{"topic": "cats"}, {"topic": "dogs"}]) }}\nTopic: {{ promptdata("topic") }}\nAlways respond: {"results": "pass"}'
NO_CONFIG_TEST = 'Always respond with exactly this JSON: {"results": "pass"}'


def _make_storage():
    s = MemoryStorageBackend()
    s.init_db()
    return s


def test_execute_tests_no_config_no_row_index() -> None:
    tests = {"smoke": NO_CONFIG_TEST}
    storage = _make_storage()
    run_id = storage.create_run(model_count=0)
    results = execute_tests(
        run_id=run_id, tests=tests, model_outputs={},
        storage_backend=storage, llm_call=PASS_LLM,
    )
    assert len(results) == 1
    assert results[0].test_name == "smoke"
    assert results[0].status == "pass"


def test_execute_tests_row_index_adds_suffix() -> None:
    tests = {"t": SIMPLE_TEST}
    storage = _make_storage()
    run_id = storage.create_run(model_count=0)
    results = execute_tests(
        run_id=run_id, tests=tests, model_outputs={},
        storage_backend=storage, llm_call=PASS_LLM, row_index=0,
    )
    assert results[0].test_name == "t[row_1]"
    assert results[0].param_label == "row_1"


def test_execute_tests_promptdata_injected_from_config() -> None:
    captured: list[str] = []

    def recording_llm(prompt: str) -> str:
        captured.append(prompt)
        return json.dumps({"results": "pass"})

    tests = {"t": SIMPLE_TEST}
    storage = _make_storage()

    run_id0 = storage.create_run(model_count=0)
    execute_tests(run_id=run_id0, tests=tests, model_outputs={},
                  storage_backend=storage, llm_call=recording_llm, row_index=0)
    assert any("cats" in p for p in captured)

    run_id1 = storage.create_run(model_count=0)
    execute_tests(run_id=run_id1, tests=tests, model_outputs={},
                  storage_backend=storage, llm_call=recording_llm, row_index=1)
    assert any("dogs" in p for p in captured)


def test_execute_tests_multiple_tests_same_row() -> None:
    src = '{{ config(promptdata=[{"k": "a"}, {"k": "b"}]) }}\nAlways respond: {"results": "pass"}'
    tests = {"alpha": src, "beta": src}
    storage = _make_storage()
    run_id = storage.create_run(model_count=0)
    results = execute_tests(
        run_id=run_id, tests=tests, model_outputs={},
        storage_backend=storage, llm_call=PASS_LLM, row_index=0,
    )
    assert len(results) == 2
    names = {r.test_name for r in results}
    assert names == {"alpha[row_1]", "beta[row_1]"}


# ---------------------------------------------------------------------------
# CLI integration: pbt test with inline config rows
# ---------------------------------------------------------------------------

SIMPLE_CLIENT_PY = """\
import json

def llm_call(prompt: str, config: dict | None = None) -> str:
    if "respond only with valid json" in prompt.lower():
        return json.dumps({"results": "pass"})
    return "Hello from " + prompt[:20]
"""

TEST_WITH_CONFIG = """\
{{ config(
    promptdata=[{"tone": "formal"}, {"tone": "casual"}]
) }}
Does the style match '{{ promptdata("tone") }}'?
Respond ONLY with valid JSON: {"results": "pass"} if yes.
"""


@pytest.fixture()
def config_proj(tmp_path: Path) -> Path:
    proj = tmp_path / "cfg_proj"
    (proj / "models").mkdir(parents=True)
    (proj / "tests").mkdir()

    (proj / "models" / "greet.prompt").write_text(
        "Say hello in one word.", encoding="utf-8"
    )
    (proj / "tests" / "tone_test.prompt").write_text(TEST_WITH_CONFIG, encoding="utf-8")
    (proj / "client.py").write_text(SIMPLE_CLIENT_PY, encoding="utf-8")

    yield proj
    shutil.rmtree(proj)


def test_cli_test_uses_config_rows(config_proj: Path) -> None:
    """pbt test with 2 config rows produces 2 test cases named test[row_N]."""
    result = run_pbt("test", cwd=config_proj, check=False)
    output = result.stdout + result.stderr
    assert "tone_test[row_1]" in output
    assert "tone_test[row_2]" in output


def test_cli_test_no_config_uses_latest_run(tmp_path: Path) -> None:
    """pbt test with no config rows tests against the latest run."""
    proj = tmp_path / "plain_proj"
    (proj / "models").mkdir(parents=True)
    (proj / "tests").mkdir()
    (proj / "models" / "greet.prompt").write_text("Say hello.", encoding="utf-8")
    (proj / "tests" / "smoke.prompt").write_text(
        'Always respond with exactly this JSON: {"results": "pass"}',
        encoding="utf-8",
    )
    (proj / "client.py").write_text(SIMPLE_CLIENT_PY, encoding="utf-8")

    run_pbt("run", cwd=proj)
    result = run_pbt("test", cwd=proj, check=False)
    output = result.stdout + result.stderr
    assert "smoke" in output
