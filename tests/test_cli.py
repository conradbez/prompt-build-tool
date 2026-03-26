"""
CLI integration tests — exercise all model types and basic model/validation/test
workflow through the real pbt binary.  Each test writes files to a tmp dir that
pytest cleans up automatically after the session.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.conftest import MODELS, STUB_CLIENT_PY, run_pbt


def _write_models(models_dir: Path, models: dict[str, str]) -> None:
    models_dir.mkdir(exist_ok=True)
    for name, source in models.items():
        (models_dir / f"{name}.prompt").write_text(source, encoding="utf-8")


# ---------------------------------------------------------------------------
# All model types
# ---------------------------------------------------------------------------

def test_cli_run_all_model_types(tmp_path: Path) -> None:
    run_pbt("init", "proj", "--provider", "openai", cwd=tmp_path)
    proj = tmp_path / "proj"

    for folder in ("models", "validation", "tests"):
        shutil.rmtree(proj / folder, ignore_errors=True)

    _write_models(proj / "models", MODELS)
    (proj / "client.py").write_text(STUB_CLIENT_PY, encoding="utf-8")

    result = run_pbt("run", cwd=proj, check=False)

    assert result.returncode == 0, (
        f"pbt run failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "errored" not in result.stdout.lower() or "ERROR=0" in result.stdout


# ---------------------------------------------------------------------------
# Simple model + validation + test workflow
# ---------------------------------------------------------------------------

SIMPLE_CLIENT_PY = """\
import json

def llm_call(prompt: str, config: dict | None = None) -> str:
    # pbt test prompts ask for a results JSON verdict
    if "respond only with valid json" in prompt.lower():
        return json.dumps({"results": "pass"})
    return "Hello"
"""

SIMPLE_VALIDATION_PY = """\
def validate(prompt: str, result: str) -> str:
    if not result.strip():
        return False
    return result
"""

SIMPLE_TEST_PROMPT = """\
Does the following contain a greeting? {{ ref('greet') }}
Respond ONLY with valid JSON: {"results": "pass"} if yes, {"results": "fail"} if no.
"""


@pytest.fixture()
def simple_proj(tmp_path: Path) -> Path:
    proj = tmp_path / "simple"
    (proj / "models").mkdir(parents=True)
    (proj / "validation").mkdir()
    (proj / "tests").mkdir()

    (proj / "models" / "greet.prompt").write_text("Say hello in one word.", encoding="utf-8")
    (proj / "validation" / "greet.py").write_text(SIMPLE_VALIDATION_PY, encoding="utf-8")
    (proj / "tests" / "greet_is_greeting.prompt").write_text(SIMPLE_TEST_PROMPT, encoding="utf-8")
    (proj / "client.py").write_text(SIMPLE_CLIENT_PY, encoding="utf-8")

    yield proj

    shutil.rmtree(proj)


def test_simple_run_succeeds(simple_proj: Path) -> None:
    result = run_pbt("run", cwd=simple_proj, check=False)
    assert result.returncode == 0, result.stderr


def test_simple_output_written(simple_proj: Path) -> None:
    run_pbt("run", cwd=simple_proj)
    assert (simple_proj / "outputs" / "greet.md").exists()


def test_simple_pbt_test_passes(simple_proj: Path) -> None:
    run_pbt("run", cwd=simple_proj)
    result = run_pbt("test", cwd=simple_proj, check=False)
    assert result.returncode == 0, result.stderr
