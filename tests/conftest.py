"""Shared helpers for pbt tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Stub LLM — covers all model types used in MODELS
# ---------------------------------------------------------------------------

def stub_llm(prompt: str, config: dict | None = None) -> str:
    """Deterministic stub that satisfies every model type in MODELS."""
    if (config or {}).get("output_format") == "json":
        return json.dumps(["item_one", "item_two"])
    if "pass if yes" in prompt.lower():
        return "PASS"
    return "mock response"


STUB_CLIENT_PY = """\
import json

def llm_call(prompt: str, config: dict | None = None) -> str:
    if (config or {}).get("output_format") == "json":
        return json.dumps(["item_one", "item_two"])
    if "pass if yes" in prompt.lower():
        return "PASS"
    return "mock response"
"""


# ---------------------------------------------------------------------------
# One of each model type — shared across both test modules
# ---------------------------------------------------------------------------

MODELS: dict[str, str] = {
    # plain LLM node, no deps
    "topic": "Name one topic. Reply with just the topic name.",

    # JSON output — required as the loop source
    "items": (
        '{{ config(output_format="json") }}\n'
        'List 2 things about: {{ ref("topic") }}\n'
        'Return a JSON array of exactly 2 strings.'
    ),

    # loop — fans out over the items list, one LLM call per item
    "items_loop": (
        '{{ config(model_type="loop") }}\n'
        'Describe this in one sentence: {{ ref("items") }}'
    ),

    # execute_python — no LLM call; reads upstream ref()
    "transform": (
        '{{ config(model_type="execute_python") }}\n'
        'output = ref("topic").upper()'
    ),

    # quality_check — expands into check + retry chain at DAG-build time
    "topic_quality": (
        '{{ config(model_type="quality_check", quality_retries="1") }}\n'
        'Is this a valid topic? {{ ref("topic") }}\n'
        'Reply PASS if yes, FAIL otherwise.'
    ),

    # terminal node that consumes all the above
    "summary": (
        'Topic: {{ ref("topic_quality") }}\n'
        'Upper: {{ ref("transform") }}\n'
        'Items: {{ ref("items_loop") }}\n'
        'Write a brief summary.'
    ),
}


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

PBT_BIN = str(Path(sys.executable).parent / "pbt")


def run_pbt(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PBT_BIN, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )
