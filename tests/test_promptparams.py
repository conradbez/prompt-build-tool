"""
Tests for promptparams.csv support — parameterised test runs.

Covers:
- load_promptparams: missing file, valid CSV
- parse_promptparams_row: promptdata/promptfile columns, JSON array paths
- write_example: generates correct headers + placeholder row
- execute_tests cross-join: test × row combinations, correct naming
- CLI pbt test: loads promptparams.csv, generates .example file
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from pbt.promptparams import (
    load_promptparams,
    parse_promptparams_row,
    write_example,
    get_promptparams_columns,
    PROMPTDATA_PREFIX,
    PROMPTFILE_PREFIX,
)
from pbt.tester import execute_tests, load_tests
from pbt.storage.memory import MemoryStorageBackend
from tests.conftest import run_pbt


# ---------------------------------------------------------------------------
# load_promptparams
# ---------------------------------------------------------------------------

def test_load_promptparams_missing_file(tmp_path: Path) -> None:
    rows = load_promptparams(tmp_path / "nonexistent.csv")
    assert rows == []


def test_load_promptparams_returns_rows(tmp_path: Path) -> None:
    csv_file = tmp_path / "promptparams.csv"
    csv_file.write_text(
        "promptdata.tone,promptdata.audience\n"
        "formal,engineers\n"
        "casual,developers\n",
        encoding="utf-8",
    )
    rows = load_promptparams(csv_file)
    assert len(rows) == 2
    assert rows[0] == {"promptdata.tone": "formal", "promptdata.audience": "engineers"}
    assert rows[1] == {"promptdata.tone": "casual", "promptdata.audience": "developers"}


# ---------------------------------------------------------------------------
# parse_promptparams_row
# ---------------------------------------------------------------------------

def test_parse_promptparams_row_promptdata() -> None:
    row = {"promptdata.tone": "formal", "promptdata.audience": "engineers"}
    pd, pf = parse_promptparams_row(row)
    assert pd == {"tone": "formal", "audience": "engineers"}
    assert pf == {}


def test_parse_promptparams_row_promptfile_single_path() -> None:
    row = {"promptfile.document": "report.pdf"}
    pd, pf = parse_promptparams_row(row)
    assert pd == {}
    assert pf == {"document": "report.pdf"}


def test_parse_promptparams_row_promptfile_json_array() -> None:
    row = {"promptfile.docs": '["a.pdf", "b.pdf"]'}
    pd, pf = parse_promptparams_row(row)
    assert pf == {"docs": ["a.pdf", "b.pdf"]}


def test_parse_promptparams_row_mixed() -> None:
    row = {
        "promptdata.tone": "formal",
        "promptfile.report": "annual.pdf",
        "promptfile.chart": '["q1.png", "q2.png"]',
    }
    pd, pf = parse_promptparams_row(row)
    assert pd == {"tone": "formal"}
    assert pf["report"] == "annual.pdf"
    assert pf["chart"] == ["q1.png", "q2.png"]


def test_parse_promptparams_row_skips_empty_cells() -> None:
    row = {"promptdata.tone": "formal", "promptdata.audience": ""}
    pd, pf = parse_promptparams_row(row)
    assert "audience" not in pd


# ---------------------------------------------------------------------------
# get_promptparams_columns
# ---------------------------------------------------------------------------

def test_get_promptparams_columns() -> None:
    rows = [
        {"promptdata.tone": "formal", "promptfile.doc": "a.pdf"},
        {"promptdata.tone": "casual", "promptfile.doc": "b.pdf"},
    ]
    pd_keys, pf_keys = get_promptparams_columns(rows)
    assert pd_keys == ["tone"]
    assert pf_keys == ["doc"]


# ---------------------------------------------------------------------------
# write_example
# ---------------------------------------------------------------------------

def test_write_example_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "promptparams.csv.example"
    write_example(out, ["tone", "audience"], ["document"])

    assert out.exists()
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == [
        "promptdata.tone",
        "promptdata.audience",
        "promptfile.document",
    ]
    # Placeholder row
    assert rows[1][0] == "<tone>"
    assert rows[1][1] == "<audience>"
    assert rows[1][2] == "<path/to/document>"


def test_write_example_no_columns_does_nothing(tmp_path: Path) -> None:
    out = tmp_path / "promptparams.csv.example"
    write_example(out, [], [])
    assert not out.exists()


def test_write_example_only_promptdata(tmp_path: Path) -> None:
    out = tmp_path / "promptparams.csv.example"
    write_example(out, ["topic"], [])
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["promptdata.topic"]


# ---------------------------------------------------------------------------
# execute_tests cross-join
# ---------------------------------------------------------------------------

def _stub_llm_pass(prompt: str) -> str:
    return json.dumps({"results": "pass"})


def _stub_llm_echo_promptdata(prompt: str) -> str:
    """Return pass; the test inspects the rendered prompt instead."""
    return json.dumps({"results": "pass"})


SIMPLE_TEST_SOURCE = """\
Topic is: {{ promptdata("topic") }}
Always respond with exactly this JSON: {"results": "pass"}
"""

NO_PARAMS_TEST_SOURCE = """\
Always respond with exactly this JSON: {"results": "pass"}
"""


def test_execute_tests_no_promptparams() -> None:
    """Without promptparams, each test runs once with its base name."""
    tests = {"smoke": NO_PARAMS_TEST_SOURCE}
    storage = MemoryStorageBackend()
    storage.init_db()
    run_id = storage.create_run(model_count=0)

    results = execute_tests(
        run_id=run_id,
        tests=tests,
        model_outputs={},
        storage_backend=storage,
        llm_call=_stub_llm_pass,
        promptparams_rows=None,
    )

    assert len(results) == 1
    assert results[0].test_name == "smoke"
    assert results[0].status == "pass"
    assert results[0].param_label == ""


def test_execute_tests_cross_join_naming() -> None:
    """With 2 rows × 2 tests → 4 results named test[row_N]."""
    tests = {
        "alpha": NO_PARAMS_TEST_SOURCE,
        "beta": NO_PARAMS_TEST_SOURCE,
    }
    rows = [
        {"promptdata.tone": "formal"},
        {"promptdata.tone": "casual"},
    ]
    storage = MemoryStorageBackend()
    storage.init_db()
    run_id = storage.create_run(model_count=0)

    results = execute_tests(
        run_id=run_id,
        tests=tests,
        model_outputs={},
        storage_backend=storage,
        llm_call=_stub_llm_pass,
        promptparams_rows=rows,
    )

    assert len(results) == 4
    names = {r.test_name for r in results}
    assert names == {"alpha[row_1]", "alpha[row_2]", "beta[row_1]", "beta[row_2]"}

    for r in results:
        assert r.param_label in ("row_1", "row_2")
        assert r.status == "pass"


def test_execute_tests_promptdata_injected() -> None:
    """promptdata from each row is rendered into the test template."""
    captured_prompts: list[str] = []

    def recording_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return json.dumps({"results": "pass"})

    tests = {"pd_test": SIMPLE_TEST_SOURCE}
    rows = [
        {"promptdata.topic": "cats"},
        {"promptdata.topic": "dogs"},
    ]
    storage = MemoryStorageBackend()
    storage.init_db()
    run_id = storage.create_run(model_count=0)

    execute_tests(
        run_id=run_id,
        tests=tests,
        model_outputs={},
        storage_backend=storage,
        llm_call=recording_llm,
        promptparams_rows=rows,
    )

    assert any("cats" in p for p in captured_prompts)
    assert any("dogs" in p for p in captured_prompts)


def test_execute_tests_single_row_suffix() -> None:
    """Single row → test name ends with [row_1]."""
    tests = {"t": NO_PARAMS_TEST_SOURCE}
    rows = [{"promptdata.x": "1"}]
    storage = MemoryStorageBackend()
    storage.init_db()
    run_id = storage.create_run(model_count=0)

    results = execute_tests(
        run_id=run_id,
        tests=tests,
        model_outputs={},
        storage_backend=storage,
        llm_call=_stub_llm_pass,
        promptparams_rows=rows,
    )

    assert results[0].test_name == "t[row_1]"
    assert results[0].param_label == "row_1"


# ---------------------------------------------------------------------------
# CLI integration: pbt test with promptparams.csv
# ---------------------------------------------------------------------------

SIMPLE_PROMPTPARAMS_CLIENT_PY = """\
import json

def llm_call(prompt: str, config: dict | None = None) -> str:
    if "respond only with valid json" in prompt.lower():
        return json.dumps({"results": "pass"})
    return "Hello"
"""

SIMPLE_TEST_PROMPT_WITH_PROMPTDATA = """\
Does the following style match 'formal'? promptdata value: {{ promptdata("tone") }}
Respond ONLY with valid JSON: {"results": "pass"} if it does, {"results": "fail"} if not.
"""


@pytest.fixture()
def promptparams_proj(tmp_path: Path) -> Path:
    proj = tmp_path / "pp_proj"
    (proj / "models").mkdir(parents=True)
    (proj / "tests").mkdir()

    (proj / "models" / "greet.prompt").write_text(
        "Say hello in one word.", encoding="utf-8"
    )
    (proj / "tests" / "tone_test.prompt").write_text(
        SIMPLE_TEST_PROMPT_WITH_PROMPTDATA, encoding="utf-8"
    )
    (proj / "client.py").write_text(SIMPLE_PROMPTPARAMS_CLIENT_PY, encoding="utf-8")

    # Write a promptparams.csv with two rows
    (proj / "promptparams.csv").write_text(
        "promptdata.tone\nformal\ncasual\n", encoding="utf-8"
    )

    yield proj
    shutil.rmtree(proj)


def test_cli_test_uses_promptparams(promptparams_proj: Path) -> None:
    """pbt run + pbt test should execute 2 test cases (1 test × 2 rows)."""
    run_pbt("run", cwd=promptparams_proj)
    result = run_pbt("test", cwd=promptparams_proj, check=False)

    # Both parameterised variants should appear in output
    assert "tone_test[row_1]" in result.stdout or "tone_test[row_1]" in result.stderr
    assert "tone_test[row_2]" in result.stdout or "tone_test[row_2]" in result.stderr


def test_cli_test_writes_example(promptparams_proj: Path) -> None:
    """pbt test should write promptparams.csv.example."""
    run_pbt("run", cwd=promptparams_proj)
    run_pbt("test", cwd=promptparams_proj, check=False)

    example = promptparams_proj / "promptparams.csv.example"
    assert example.exists(), "promptparams.csv.example was not created"

    with example.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # Should have a header row
    assert len(rows) >= 1
    headers = rows[0]
    # At least one promptdata column (tone is used in the test prompt)
    assert any(h.startswith("promptdata.") for h in headers)
