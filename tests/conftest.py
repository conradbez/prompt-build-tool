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
    return "mock response"


STUB_CLIENT_PY = """\
import json

def llm_call(prompt: str, config: dict | None = None) -> str:
    if (config or {}).get("output_format") == "json":
        return json.dumps(["item_one", "item_two"])
    return "mock response"
"""


# ---------------------------------------------------------------------------
# One of each model type — shared across both test modules
# ---------------------------------------------------------------------------

MODELS: dict[str, str] = {
    # plain LLM node, no deps
    "topic": "Name one topic. Reply with just the topic name.",

    # JSON output
    "items": (
        '{{ config(output_format="json") }}\n'
        'List 2 things about: {{ ref("topic") }}\n'
        'Return a JSON array of exactly 2 strings.'
    ),

    # execute_python — no LLM call; reads upstream ref()
    "transform": (
        '{{ config(model_type="execute_python") }}\n'
        'output = ref("topic").upper()'
    ),

    # template — no LLM call; the rendered text is the output
    "header": (
        '{{ config(model_type="template") }}\n'
        '# {{ ref("topic") }}'
    ),

    # terminal node that consumes all the above
    "summary": (
        'Topic: {{ ref("header") }}\n'
        'Upper: {{ ref("transform") }}\n'
        'Items: {{ ref("items") }}\n'
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
