"""
Integration test: JSON output model feeding a loop model via a real LLM call.

Requires GEMINI_API_KEY to be set; skipped otherwise.

The two models under tests/test_integration_json_loop/models/ are:
  list_items.prompt  — config(output_format="json"), returns a JSON array
  describe_items.prompt — config(model_type="loop"), fans out over that array

This test verifies the full round-trip: the LLM produces a valid JSON array,
the executor parses it, and the loop model receives each item individually and
returns one response per item.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import pbt
from pbt.storage import MemoryStorageBackend

MODELS_DIR = str(Path(__file__).parent / "test_integration_json_loop" / "models")

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — skipping live LLM integration test",
)


def _make_gemini_llm_call():
    from google import genai

    api_key = os.environ["GEMINI_API_KEY"]
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    client = genai.Client(api_key=api_key)

    def llm_call(prompt: str, config: dict | None = None) -> str:
        return client.models.generate_content(model=model_name, contents=prompt).text

    return llm_call


async def test_json_template_feeds_loop_model() -> None:
    """list_items returns a parsed JSON list; describe_items loops over it."""
    llm_call = _make_gemini_llm_call()

    outputs = await pbt.async_run(
        models_dir=MODELS_DIR,
        llm_call=llm_call,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
        validation_dir=None,
    )

    # list_items: LLM returned JSON, executor parsed it — should be a list stored as JSON string
    assert "list_items" in outputs, f"list_items missing from outputs: {list(outputs)}"
    assert not isinstance(outputs["list_items"], pbt.ModelError), f"list_items errored: {outputs['list_items']}"

    list_items_raw = outputs["list_items"]
    parsed_list = json.loads(list_items_raw)
    assert isinstance(parsed_list, list), f"list_items output is not a JSON array: {list_items_raw!r}"
    assert len(parsed_list) > 0, "list_items returned an empty array"

    # describe_items: loop model ran once per item — output is a JSON list of responses
    assert "describe_items" in outputs, f"describe_items missing from outputs: {list(outputs)}"
    assert not isinstance(outputs["describe_items"], pbt.ModelError), f"describe_items errored: {outputs['describe_items']}"

    describe_raw = outputs["describe_items"]
    describe_list = json.loads(describe_raw)
    assert isinstance(describe_list, list), f"describe_items output is not a list: {describe_raw!r}"
    assert len(describe_list) == len(parsed_list), (
        f"describe_items produced {len(describe_list)} results but list_items had {len(parsed_list)} items"
    )

    for i, item_response in enumerate(describe_list):
        assert isinstance(item_response, str) and item_response.strip(), (
            f"describe_items[{i}] is empty or not a string: {item_response!r}"
        )
