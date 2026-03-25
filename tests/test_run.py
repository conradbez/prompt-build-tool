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


# def test_run_with_file_upload(tmp_path: Path) -> None:
#     """A prompt that declares promptfiles= receives the file and runs successfully."""
#     proj = init_project(tmp_path)
#     (proj / "client.py").write_text(STUB_CLIENT_FILES_PY)

#     # Write a standalone prompt that declares a file dependency
#     (proj / "models" / "summarise_doc.prompt").write_text(
#         '{{ config(promptfiles="doc") }}\n'
#         "Summarise the contents of the uploaded document.\n"
#     )

#     # Generate the txt file to upload
#     doc = tmp_path / "sample.txt"
#     doc.write_text("This is a sample document used in the pbt file-upload test.", encoding="utf-8")

#     result = run_pbt(
#         "run", "--select", "summarise_doc",
#         "--promptfile", f"doc={doc}",
#         cwd=proj,
#     )
#     assert result.returncode == 0
#     assert "succeeded" in result.stdout


# ---------------------------------------------------------------------------
# Loop model tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_model_calls_llm_per_item(tmp_path: Path) -> None:
    """Loop model iterates over upstream list and combines results."""
    import json
    from pbt.storage import MemoryStorageBackend

    models = {
        "items": '{{ config(output_format="json") }}\nReturn a list.',
        "processed": '{{ config(model_type="loop") }}\nProcess: {{ ref("items") }}',
    }

    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if "Return a list." in prompt:
            return json.dumps(["apple", "banana", "cherry"])
        return f"processed: {prompt.split('Process: ')[-1].strip()}"

    outputs = await pbt.run(
        models_from_dict=models,
        llm_call=fake_llm,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert "processed" in outputs
    assert not isinstance(outputs["processed"], pbt.ModelStatus)
    output = json.loads(outputs["processed"])
    assert isinstance(output, list)
    assert len(output) == 3
    # LLM was called once for items + 3 times for each element
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_loop_model_output_available_downstream() -> None:
    """Downstream model can ref() the loop model's combined list output."""
    import json
    from pbt.storage import MemoryStorageBackend

    models = {
        "items": '{{ config(output_format="json") }}\nList.',
        "processed": '{{ config(model_type="loop") }}\nItem: {{ ref("items") }}',
        "summary": 'Summarize: {{ ref("processed") }}',
    }

    def fake_llm(prompt: str) -> str:
        if "List." in prompt:
            return json.dumps(["x", "y"])
        if "Item:" in prompt:
            return f"done:{prompt.split('Item:')[-1].strip()}"
        return f"summary of {prompt}"

    outputs = await pbt.run(
        models_from_dict=models,
        llm_call=fake_llm,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert "summary" in outputs
    assert not isinstance(outputs["summary"], pbt.ModelStatus)
    # The summary prompt contained the combined loop output (a JSON list)
    assert "done:" in outputs["summary"]


@pytest.mark.asyncio
async def test_loop_model_error_when_no_list_dep() -> None:
    """Loop model errors clearly when no upstream returns a list."""
    from pbt.storage import MemoryStorageBackend

    models = {
        "items": "Just a text model.",
        "processed": '{{ config(model_type="loop") }}\nItem: {{ ref("items") }}',
    }

    def fake_llm(prompt: str) -> str:
        return "plain text, not a list"

    outputs = await pbt.run(
        models_from_dict=models,
        llm_call=fake_llm,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert outputs["processed"] == pbt.ModelStatus.ERROR


# ---------------------------------------------------------------------------
# execute_python model tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_python_basic_stdout() -> None:
    """execute_python model captures stdout as output."""
    from pbt.storage import MemoryStorageBackend

    models = {
        "calc": '{{ config(model_type="execute_python") }}\nprint(1 + 1)',
    }

    outputs = await pbt.async_run(
        models_from_dict=models,
        llm_call=lambda p: "unused",
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert outputs["calc"] == "2"


@pytest.mark.asyncio
async def test_execute_python_output_variable() -> None:
    """execute_python model falls back to `output` variable when stdout is empty."""
    from pbt.storage import MemoryStorageBackend

    models = {
        "calc": '{{ config(model_type="execute_python") }}\noutput = 6 * 7',
    }

    outputs = await pbt.async_run(
        models_from_dict=models,
        llm_call=lambda p: "unused",
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert outputs["calc"] == "42"


@pytest.mark.asyncio
async def test_execute_python_json_output_format() -> None:
    """execute_python model with output_format='json' parses output as JSON."""
    import json
    from pbt.storage import MemoryStorageBackend

    models = {
        "data": '{{ config(model_type="execute_python", output_format="json") }}\nprint(\'["a", "b", "c"]\')',
    }

    outputs = await pbt.async_run(
        models_from_dict=models,
        llm_call=lambda p: "unused",
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert json.loads(outputs["data"]) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_execute_python_accesses_upstream_ref() -> None:
    """execute_python model can read upstream model outputs via ref()."""
    from pbt.storage import MemoryStorageBackend

    models = {
        "name": "Just return a name.",
        "greeting": '{{ config(model_type="execute_python") }}\nupstream = ref("name")\nprint(f"Hello, {upstream}!")',
    }

    def fake_llm(prompt: str) -> str:
        return "World"

    outputs = await pbt.async_run(
        models_from_dict=models,
        llm_call=fake_llm,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert outputs["greeting"] == "Hello, World!"


@pytest.mark.asyncio
async def test_execute_python_syntax_error_marks_model_error() -> None:
    """execute_python model reports an error on invalid Python."""
    from pbt.storage import MemoryStorageBackend

    models = {
        "bad": '{{ config(model_type="execute_python") }}\ndef (bad syntax',
    }

    outputs = await pbt.async_run(
        models_from_dict=models,
        llm_call=lambda p: "unused",
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert isinstance(outputs["bad"], pbt.ModelError)


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


@pytest.mark.asyncio
async def test_python_api_returns_skip_and_set_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proj = init_project(tmp_path)
    (proj / "models" / "articles.prompt").write_text(
        '{{ skip_and_set_to_value("precomputed value") }}\n',
        encoding="utf-8",
    )

    monkeypatch.chdir(proj)
    result = await pbt.run(models_dir="models", verbose=False)

    assert result["articles"] == "precomputed value"


@pytest.mark.asyncio
async def test_python_api_renders_jinja_inside_skip_and_set_value() -> None:
    def llm_call(prompt: str) -> str:
        raise AssertionError("llm_call should not run when skip_and_set_to_value is used")

    result = await pbt.run(
        models_from_dict={
            "topic": '{{ skip_and_set_to_value("computed {{ promptdata(\'subject\') }}") }}',
        },
        llm_call=llm_call,
        promptdata={"subject": "value"},
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert result["topic"] == "computed value"


@pytest.mark.asyncio
async def test_python_api_inline_models_support_memory_storage_without_disk_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
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

    result = await pbt.run(
        models_from_dict={"topic": '{{ skip_and_set_to_value("browser-safe") }}'},
        llm_call=lambda prompt, **kwargs: prompt,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert result["topic"] == "browser-safe"


@pytest.mark.asyncio
async def test_python_api_supports_async_llm_call() -> None:
    async def llm_call(prompt: str) -> str:
        return "ok"

    result = await pbt.run(
        models_from_dict={"topic": "Return ok"},
        llm_call=llm_call,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert result["topic"] == "ok"


@pytest.mark.asyncio
async def test_skip_this_and_downstream_skips_current_and_children() -> None:
    """skip_this_and_downstream skips the caller and all downstream models."""
    llm_called_for: list[str] = []

    def llm_call(prompt: str) -> str:
        llm_called_for.append(prompt)
        return "should not reach"

    result = await pbt.run(
        models_from_dict={
            "gate": '{{ skip_this_and_downstream("skipped gate") }}',
            "child": "{{ ref('gate') }} — do something",
            "grandchild": "{{ ref('child') }} — do more",
        },
        llm_call=llm_call,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert llm_called_for == [], "LLM should not be called for any model"
    assert result["gate"] == "skipped gate"
    assert result["child"] is pbt.ModelStatus.SKIPPED
    assert result["grandchild"] is pbt.ModelStatus.SKIPPED


@pytest.mark.asyncio
async def test_skip_this_and_downstream_does_not_affect_unrelated_models() -> None:
    """skip_this_and_downstream only skips the signalling branch, not unrelated models."""
    def llm_call(prompt: str) -> str:
        return "ran"

    result = await pbt.run(
        models_from_dict={
            "gate": '{{ skip_this_and_downstream("") }}',
            "child": "{{ ref('gate') }} — skipped branch",
            "independent": "independent model with no deps",
        },
        llm_call=llm_call,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert result["independent"] == "ran"
    assert result["child"] is pbt.ModelStatus.SKIPPED


@pytest.mark.asyncio
async def test_python_api_supports_sync_llm_call_under_async_run() -> None:
    def llm_call(prompt: str) -> str:
        return "ok"

    result = await pbt.run(
        models_from_dict={"topic": "Return ok"},
        llm_call=llm_call,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    assert result["topic"] == "ok"
