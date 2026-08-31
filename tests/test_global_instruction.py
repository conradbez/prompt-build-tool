"""
Global instruction — pbt's analogue of dbt's query-comment.

Covers the two sources (kwarg / global.prompt), the two placements
(prepend / {{ prompt }} wrapper), the opt-outs, and cache invalidation.
"""

from __future__ import annotations

import pytest

import pbt
from pbt.storage import MemoryStorageBackend

from tests.conftest import STUB_CLIENT_PY, run_pbt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RecordingLLM:
    """Stub llm_call that records every prompt it is handed."""

    def __init__(self, response: str = "mock response") -> None:
        self.prompts: list[str] = []
        self.response = response

    def __call__(self, prompt: str, config: dict | None = None) -> str:
        self.prompts.append(prompt)
        if (config or {}).get("output_format") == "json":
            return '["a", "b"]'
        return self.response


async def _run(models: dict[str, str], llm, **kwargs) -> dict:
    return await pbt.async_run(
        models_from_dict=models,
        llm_call=llm,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
        **kwargs,
    )


SIMPLE = {"topic": "Name one topic."}


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

async def test_prepended_by_default() -> None:
    llm = RecordingLLM()
    await _run(SIMPLE, llm, global_instruction="Always answer in French.")
    assert llm.prompts == ["Always answer in French.\n\nName one topic."]


async def test_prompt_placeholder_wraps_the_body() -> None:
    llm = RecordingLLM()
    await _run(SIMPLE, llm, global_instruction="<start>\n{{ prompt }}\n<end>")
    assert llm.prompts == ["<start>\nName one topic.\n<end>"]


async def test_no_global_instruction_leaves_prompt_untouched() -> None:
    llm = RecordingLLM()
    await _run(SIMPLE, llm)
    assert llm.prompts == ["Name one topic."]


async def test_instruction_rendering_to_blank_leaves_prompt_untouched() -> None:
    llm = RecordingLLM()
    await _run(SIMPLE, llm, global_instruction='{% if promptdata("x") %}hi{% endif %}')
    assert llm.prompts == ["Name one topic."]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

async def test_callable_source_is_invoked() -> None:
    llm = RecordingLLM()
    await _run(SIMPLE, llm, global_instruction=lambda: "From a callable.")
    assert llm.prompts[0].startswith("From a callable.")


async def test_callable_is_invoked_once_for_the_whole_run() -> None:
    calls = []

    def build() -> str:
        calls.append(1)
        return "Once."

    llm = RecordingLLM()
    await _run({"a": "A.", "b": 'B: {{ ref("a") }}'}, llm, global_instruction=build)
    assert len(calls) == 1


async def test_bad_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="global_instruction must be"):
        await _run(SIMPLE, RecordingLLM(), global_instruction=42)


# ---------------------------------------------------------------------------
# Jinja context
# ---------------------------------------------------------------------------

async def test_promptdata_and_model_name_available() -> None:
    llm = RecordingLLM()
    await _run(
        SIMPLE, llm,
        promptdata={"tone": "formal"},
        global_instruction='Tone: {{ promptdata("tone") }} for {{ model.name }}.',
    )
    assert llm.prompts[0].startswith("Tone: formal for topic.")


async def test_ref_is_rejected_with_a_clear_error() -> None:
    llm = RecordingLLM()
    outputs = await _run(SIMPLE, llm, global_instruction='{{ ref("topic") }}')
    assert isinstance(outputs["topic"], pbt.ModelError)
    assert "not available in a global instruction" in str(outputs["topic"])


# ---------------------------------------------------------------------------
# Opt-outs
# ---------------------------------------------------------------------------

async def test_model_opts_out_via_config() -> None:
    models = {
        "plain": "Plain body.",
        "optout": '{{ config(global_instruction=False) }}\nOpted out.',
    }
    llm = RecordingLLM()
    await _run(models, llm, global_instruction="GLOBAL.")
    # The config() line itself renders to "", leaving a leading newline.
    assert sorted(llm.prompts) == ["\nOpted out.", "GLOBAL.\n\nPlain body."]


async def test_execute_python_never_receives_it() -> None:
    models = {
        "topic": "Name one topic.",
        "transform": '{{ config(model_type="execute_python") }}\noutput = ref("topic").upper()',
    }
    llm = RecordingLLM()
    outputs = await _run(models, llm, global_instruction="Always answer in French.")
    # Prose prepended to Python source would be a SyntaxError.
    assert outputs["transform"] == "MOCK RESPONSE"


def test_config_key_does_not_warn() -> None:
    import warnings

    from pbt.executor.graph import build_models_from_dict
    from pbt.executor.parser_initial import UnknownConfigKeyWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_models_from_dict({"m": '{{ config(global_instruction=False) }}\nBody.'})

    offenders = [w for w in caught if issubclass(w.category, UnknownConfigKeyWarning)]
    assert not offenders, [str(w.message) for w in offenders]


# ---------------------------------------------------------------------------
# Loop models
# ---------------------------------------------------------------------------

async def test_loop_model_gets_it_per_item() -> None:
    models = {
        "items": '{{ config(output_format="json") }}\nList things.',
        "items_loop": '{{ config(model_type="loop") }}\nDescribe: {{ ref("items") }}',
    }
    llm = RecordingLLM()
    await _run(models, llm, global_instruction="GLOBAL.")
    loop_prompts = [
        p for p in llm.prompts
        if p.startswith("GLOBAL.\n\n") and "Describe:" in p
    ]
    assert len(loop_prompts) == 2, llm.prompts


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

async def test_changing_the_instruction_busts_the_cache() -> None:
    storage = MemoryStorageBackend()

    async def once(instruction: str, llm) -> None:
        await pbt.async_run(
            models_from_dict=SIMPLE, llm_call=llm, verbose=False,
            storage_backend=storage, global_instruction=instruction,
        )

    llm = RecordingLLM()
    await once("First.", llm)
    await once("First.", llm)
    assert len(llm.prompts) == 1, "identical instruction should hit the cache"

    await once("Second.", llm)
    assert len(llm.prompts) == 2, "changed instruction should miss the cache"


# ---------------------------------------------------------------------------
# File discovery (end to end, through the CLI)
# ---------------------------------------------------------------------------

def test_global_prompt_file_is_discovered(tmp_path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "topic.prompt").write_text("Name one topic.")
    (tmp_path / "client.py").write_text(STUB_CLIENT_PY)
    (tmp_path / "global.prompt").write_text("Always answer in French.")

    result = run_pbt("run", cwd=tmp_path)
    assert "Global instruction: global.prompt" in result.stdout

    stored = (tmp_path / "outputs" / "topic.md").read_text()
    assert stored  # ran to completion


def test_cli_flag_overrides_the_file(tmp_path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "topic.prompt").write_text("Name one topic.")
    (tmp_path / "client.py").write_text(STUB_CLIENT_PY)
    (tmp_path / "global.prompt").write_text("From the file.")
    (tmp_path / "override.prompt").write_text("From the flag.")

    result = run_pbt("run", "--global-instruction", "override.prompt", cwd=tmp_path)
    assert "Global instruction: override.prompt" in result.stdout
