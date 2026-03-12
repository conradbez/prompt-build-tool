"""Tests for `pbt run`."""

import textwrap
import sys
import types
from pathlib import Path

import pytest
import pbt
from pbt.storage import MemoryStorageBackend

from tests.conftest import run_pbt, init_project, init_project_with_real_client, STUB_CLIENT_JSON_PY, STUB_CLIENT_FILES_PY


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_succeeds(proj: Path) -> None:
    result = run_pbt("run", cwd=proj)
    assert result.returncode == 0
    assert "succeeded" in result.stdout


def test_run_creates_output_files(proj: Path) -> None:
    run_pbt("run", cwd=proj)
    outputs = list((proj / "outputs").glob("*.*"))
    assert len(outputs) >= 2, "Expected output files for articles and summaries"


def test_run_all_models_succeed(proj: Path) -> None:
    result = run_pbt("run", cwd=proj)
    assert "errored" not in result.stdout
    assert "skipped" not in result.stdout


def test_run_with_promptdata(proj: Path) -> None:
    result = run_pbt("run", "--promptdata", "topic=quantum computing", cwd=proj)
    assert result.returncode == 0
    assert "succeeded" in result.stdout


def test_run_select_single_model(proj: Path) -> None:
    # First full run to populate cache
    run_pbt("run", cwd=proj)
    # Select only summaries (depends on articles)
    result = run_pbt("run", "--select", "summaries", cwd=proj)
    assert result.returncode == 0
    assert "succeeded" in result.stdout


def test_run_select_unknown_model_errors(proj: Path) -> None:
    result = run_pbt("run", "--select", "nonexistent_model", cwd=proj, check=False)
    assert result.returncode != 0


def test_run_caches_identical_prompts(proj: Path) -> None:
    run_pbt("run", cwd=proj)
    result = run_pbt("run", cwd=proj)
    # Second run should show 0 ms times (all cached)
    assert "(0 ms)" in result.stdout or "0ms" in result.stdout


def test_run_json_output_format(tmp_path: Path) -> None:
    proj = init_project(tmp_path)
    # Override summaries to declare json output and use a JSON-returning stub
    (proj / "models" / "summaries.prompt").write_text(
        '{{ config(output_format="json") }}\n'
        'Return JSON {"title": "t", "summary": "s"}.\n'
        "{{ ref('articles') }}\n"
    )
    (proj / "client.py").write_text(STUB_CLIENT_JSON_PY)
    result = run_pbt("run", cwd=proj)
    assert result.returncode == 0


def test_run_stores_results_in_db(proj: Path) -> None:
    run_pbt("run", cwd=proj)
    assert (proj / ".pbt" / "pbt.db").exists()


def test_run_invalid_promptdata_format_errors(proj: Path) -> None:
    result = run_pbt("run", "--promptdata", "badformat", cwd=proj, check=False)
    assert result.returncode != 0


def test_run_validation_failure_marks_model_error(tmp_path: Path) -> None:
    proj = init_project(tmp_path)
    # Write a validation that always fails
    (proj / "validation" / "articles.py").write_text(
        "def validate(prompt: str, result: str) -> bool:\n    return False\n"
    )
    result = run_pbt("run", cwd=proj, check=False)
    assert "errored" in result.stdout or result.returncode != 0


def test_run_with_file_upload(tmp_path: Path) -> None:
    """A prompt that declares promptfiles= receives the file and runs successfully."""
    proj = init_project(tmp_path)
    (proj / "client.py").write_text(STUB_CLIENT_FILES_PY)

    # Write a standalone prompt that declares a file dependency
    (proj / "models" / "summarise_doc.prompt").write_text(
        '{{ config(promptfiles="doc") }}\n'
        "Summarise the contents of the uploaded document.\n"
    )

    # Generate the txt file to upload
    doc = tmp_path / "sample.txt"
    doc.write_text("This is a sample document used in the pbt file-upload test.", encoding="utf-8")

    result = run_pbt(
        "run", "--select", "summarise_doc",
        "--promptfile", f"doc={doc}",
        cwd=proj,
    )
    assert result.returncode == 0
    assert "succeeded" in result.stdout


def _load_env() -> dict:
    """Load .env from project root, merged with os.environ."""
    import os
    from dotenv import dotenv_values
    env_file = Path(__file__).parent.parent / ".env"
    return {**os.environ, **(dotenv_values(env_file) if env_file.exists() else {})}


def test_default_client_file_upload_gemini(tmp_path: Path) -> None:
    """Integration test: scaffolded Gemini client uploads a .txt file end-to-end."""
    env = _load_env()
    if not env.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not found in .env or environment")

    proj = init_project_with_real_client(tmp_path, provider="gemini")
    (proj / "models" / "summarise_doc.prompt").write_text(
        '{{ config(promptfiles="doc") }}\n'
        "In one sentence, what is this document about?\n",
        encoding="utf-8",
    )
    doc = tmp_path / "sample.txt"
    doc.write_text("The quick brown fox jumped over the lazy dog.", encoding="utf-8")

    result = run_pbt(
        "run", "--select", "summarise_doc", "--promptfile", f"doc={doc}",
        cwd=proj, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "succeeded" in result.stdout


def test_default_client_file_upload_openai(tmp_path: Path) -> None:
    """Integration test: scaffolded OpenAI client uploads a .txt file end-to-end."""
    env = _load_env()
    if not env.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not found in .env or environment")

    proj = init_project_with_real_client(tmp_path, provider="openai")
    (proj / "models" / "summarise_doc.prompt").write_text(
        '{{ config(promptfiles="doc") }}\n'
        "In one sentence, what is this document about?\n",
        encoding="utf-8",
    )
    doc = tmp_path / "sample.txt"
    doc.write_text("The quick brown fox jumped over the lazy dog.", encoding="utf-8")

    result = run_pbt(
        "run", "--select", "summarise_doc", "--promptfile", f"doc={doc}",
        cwd=proj, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "succeeded" in result.stdout


def test_run_downstream_skipped_on_error(tmp_path: Path) -> None:
    proj = init_project(tmp_path)
    # Make articles always fail validation — summaries depends on articles
    (proj / "validation" / "articles.py").write_text(
        "def validate(prompt: str, result: str) -> bool:\n    return False\n"
    )
    result = run_pbt("run", cwd=proj, check=False)
    assert "skipped" in result.stdout


def test_run_skip_and_set_to_value_does_not_crash(tmp_path: Path) -> None:
    proj = init_project(tmp_path)
    (proj / "models" / "articles.prompt").write_text(
        '{{ skip_and_set_to_value("# Precomputed article\\n\\nThis output bypasses the LLM call entirely.") }}\n',
        encoding="utf-8",
    )

    result = run_pbt("run", "--select", "articles", cwd=proj)

    assert result.returncode == 0, result.stderr
    assert (proj / "outputs" / "articles.md").read_text(encoding="utf-8").startswith("# Precomputed article")


def test_python_api_returns_skip_and_set_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proj = init_project(tmp_path)
    (proj / "models" / "articles.prompt").write_text(
        '{{ skip_and_set_to_value("precomputed value") }}\n',
        encoding="utf-8",
    )

    monkeypatch.chdir(proj)
    result = pbt.run(models_dir="models", verbose=False)

    assert result["articles"] == "precomputed value"


def test_python_api_inline_models_support_memory_storage_without_disk_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    failing_llm_module = types.ModuleType("pbt.llm")
    failing_rag_module = types.ModuleType("pbt.rag")
    failing_validator_module = types.ModuleType("pbt.validator")

    def _fail(*args, **kwargs):
        raise AssertionError("disk-backed resolver should not be used for inline browser-safe runs")

    failing_llm_module.resolve_llm_call = _fail
    failing_rag_module.resolve_rag_call = _fail
    failing_validator_module.load_validators = _fail

    monkeypatch.setitem(sys.modules, "pbt.llm", failing_llm_module)
    monkeypatch.setitem(sys.modules, "pbt.rag", failing_rag_module)
    monkeypatch.setitem(sys.modules, "pbt.validator", failing_validator_module)

    result = pbt.run(
        models_from_dict={"topic": '{{ skip_and_set_to_value("browser-safe") }}'},
        llm_call=lambda prompt, **kwargs: prompt,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert result["topic"] == "browser-safe"
