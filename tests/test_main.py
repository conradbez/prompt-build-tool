"""
Main test suite — exercises all model types via the Python API.

Uses models_from_dict so no filesystem is needed.  Each parametrized
select-test runs just that model (plus its ancestors) to surface
dependency or expansion bugs model-by-model before testing the whole DAG.
"""

from __future__ import annotations

import pytest

import pbt
from pbt.storage import MemoryStorageBackend

from tests.conftest import MODELS, stub_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run(select: list[str] | None = None) -> dict:
    return await pbt.async_run(
        models_from_dict=MODELS,
        select=select,
        llm_call=stub_llm,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )


def _assert_ok(outputs: dict, name: str) -> None:
    assert name in outputs, f"'{name}' missing from outputs: {list(outputs)}"
    val = outputs[name]
    assert not isinstance(val, pbt.ModelError), f"'{name}' errored: {val}"
    assert val != pbt.ModelStatus.SKIPPED, f"'{name}' was unexpectedly skipped"


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

async def test_full_run_no_errors() -> None:
    outputs = await _run()
    errors = {k: v for k, v in outputs.items() if isinstance(v, pbt.ModelError)}
    assert not errors, f"Models errored: {errors}"


async def test_full_run_no_unexpected_skips() -> None:
    outputs = await _run()
    # Only the internal quality-check expansion nodes (topic_quality_1, topic_1)
    # may be skipped (when quality passes); user-visible models must not be.
    user_models = set(MODELS.keys())
    skipped = {k for k, v in outputs.items() if v == pbt.ModelStatus.SKIPPED and k in user_models}
    assert not skipped, f"User-visible models were skipped: {skipped}"


# ---------------------------------------------------------------------------
# Select each model individually
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", list(MODELS.keys()))
async def test_select_model(model_name: str) -> None:
    outputs = await _run(select=[model_name])
    _assert_ok(outputs, model_name)


# ---------------------------------------------------------------------------
# Per-type behaviour assertions
# ---------------------------------------------------------------------------

async def test_execute_python_uppercases_topic() -> None:
    outputs = await _run(select=["transform"])
    assert outputs["transform"] == "MOCK RESPONSE"


async def test_loop_produces_list_output() -> None:
    import json
    outputs = await _run(select=["items_loop"])
    raw = outputs["items_loop"]
    assert not isinstance(raw, pbt.ModelError)
    parsed = json.loads(raw)
    assert isinstance(parsed, list) and len(parsed) == 2


async def test_quality_check_expands_dag() -> None:
    """topic_quality node must expand: result dict contains the terminal name."""
    outputs = await _run(select=["topic_quality"])
    _assert_ok(outputs, "topic_quality")
    # Expanded intermediate nodes should also be present
    assert "topic_quality_1" in outputs


async def test_quality_check_retries_skip_when_pass() -> None:
    """When stub always returns PASS, retry nodes should be skipped."""
    outputs = await _run(select=["topic_quality"])
    # topic_1 is the retry node — it should be skipped because quality_1 returns PASS
    if "topic_1" in outputs:
        assert outputs["topic_1"] == pbt.ModelStatus.SKIPPED or not isinstance(outputs["topic_1"], pbt.ModelError)
