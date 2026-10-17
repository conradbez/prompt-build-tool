"""Run-scoped secrets: reach kinds and llm_call, never the cache, storage or a repr."""

from __future__ import annotations

import asyncio
import pickle

import pytest
from click.testing import CliRunner

import promptmap
from promptmap.cli import main
from promptmap.executor.executor import execute_run
from promptmap.executor.graph import build_models_from_dict
from promptmap.executor.run_context import RunContext
from promptmap.model_types import ModelCall, _REGISTRY
from promptmap.secrets import MissingSecretsError, Secrets, parse_secret_flags
from promptmap.storage.memory import MemoryStorageBackend
from promptmap.storage.sqlite import SQLiteStorageBackend

KEY = "sk-live-0123456789abcdef"


@pytest.fixture
def kind_reading_secret():
    """A kind whose output is derived from — but never equal to — a secret."""
    seen: list[str] = []

    @promptmap.model_kind("needs_key_test")
    async def needs_key(rendered, call):
        seen.append(call.secrets["API_KEY"])
        return await call.llm(rendered)

    yield seen
    _REGISTRY.pop("needs_key_test", None)


def run_models(models, *, storage=None, llm_call=lambda prompt: "resp", **kwargs):
    storage = storage or MemoryStorageBackend()
    storage.init_db()
    specs = list(build_models_from_dict(models).values())
    run_id = storage.create_run(model_count=len(specs))
    results = asyncio.run(execute_run(
        run_id=run_id, ordered_models=specs, storage_backend=storage,
        llm_call=llm_call, **kwargs,
    ))
    storage.finish_run(run_id, "success")
    return storage, {r.model_name: r for r in results}


# ---------------------------------------------------------------------------
# The Secrets type
# ---------------------------------------------------------------------------

def test_repr_names_keys_but_never_values():
    secrets = Secrets({"API_KEY": KEY})
    assert "API_KEY" in repr(secrets)
    assert KEY not in repr(secrets)
    assert KEY not in str(secrets)


def test_unknown_name_raises_keyerror_naming_the_flag():
    with pytest.raises(KeyError, match="--secret MISSING"):
        Secrets({})["MISSING"]


def test_cannot_be_pickled():
    with pytest.raises(TypeError):
        pickle.dumps(Secrets({"API_KEY": KEY}))


def test_empty_values_are_dropped():
    assert "EMPTY" not in Secrets({"EMPTY": ""})


# ---------------------------------------------------------------------------
# Reaching kinds and llm_call
# ---------------------------------------------------------------------------

def test_kind_reads_call_secrets(kind_reading_secret):
    _, results = run_models(
        {"m": '{{ config(model_type="needs_key_test") }}\nhi'},
        secrets={"API_KEY": KEY},
    )
    assert results["m"].status == "success"
    assert kind_reading_secret == [KEY]


def test_kind_reading_unpassed_secret_fails_naming_the_flag(kind_reading_secret):
    _, results = run_models({"m": '{{ config(model_type="needs_key_test") }}\nhi'})
    assert results["m"].status == "error"
    assert "--secret API_KEY" in results["m"].error


def test_llm_call_with_secrets_parameter_receives_them():
    received = []

    def llm_call(prompt, secrets):
        received.append(secrets["API_KEY"])
        return "resp"

    _, results = run_models({"m": "hi"}, llm_call=llm_call, secrets={"API_KEY": KEY})
    assert results["m"].status == "success"
    assert received == [KEY]


def test_llm_call_without_secrets_parameter_still_works():
    _, results = run_models({"m": "hi"}, llm_call=lambda prompt: "resp", secrets={"API_KEY": KEY})
    assert results["m"].llm_output == "resp"


def test_concurrent_runs_each_see_only_their_own_secrets():
    seen: dict[str, str] = {}

    async def llm_call(prompt, secrets):
        await asyncio.sleep(0.01)  # let the two runs interleave
        seen[prompt.strip()] = secrets["API_KEY"]
        return "resp"

    async def both():
        await asyncio.gather(
            promptmap.async_run(models_from_dict={"m": "alice"}, llm_call=llm_call, verbose=False,
                          storage_backend=MemoryStorageBackend(), secrets={"API_KEY": "key-a"}),
            promptmap.async_run(models_from_dict={"m": "bob"}, llm_call=llm_call, verbose=False,
                          storage_backend=MemoryStorageBackend(), secrets={"API_KEY": "key-b"}),
        )

    asyncio.run(both())
    assert seen == {"alice": "key-a", "bob": "key-b"}


# ---------------------------------------------------------------------------
# Kept out of the cache, storage and reprs
# ---------------------------------------------------------------------------

def test_changing_a_secret_value_still_hits_the_cache():
    storage = MemoryStorageBackend()
    run_models({"m": "hi"}, storage=storage, secrets={"API_KEY": "old"})
    _, results = run_models({"m": "hi"}, storage=storage, secrets={"API_KEY": "rotated"})
    assert results["m"].cached


def test_repr_of_call_and_context_hide_the_value():
    secrets = Secrets({"API_KEY": KEY})
    ctx = RunContext(run_id="r", storage=MemoryStorageBackend(), llm_call=lambda p: p, secrets=secrets)
    call = ModelCall(spec=None, outputs={}, llm=None, compute=None, secrets=secrets)
    assert KEY not in repr(ctx)
    assert KEY not in repr(call)


def test_stored_run_never_contains_the_value(tmp_path, kind_reading_secret):
    def llm_call(prompt, secrets, config=None):
        return "resp"

    storage = SQLiteStorageBackend(tmp_path / "promptmap.db")
    run_models(
        {"m": '{{ config(model_type="needs_key_test") }}\nhi', "n": "{{ ref('m') }} again"},
        storage=storage, llm_call=llm_call, secrets={"API_KEY": KEY},
    )
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert files
    for path in files:
        assert KEY.encode() not in path.read_bytes(), path


# ---------------------------------------------------------------------------
# Kinds declaring what they need
# ---------------------------------------------------------------------------

def test_kind_declared_secrets_fail_before_any_model_runs():
    calls = []

    @promptmap.model_kind("modal_test", secrets={"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"})
    async def modal(rendered, call):
        return "done"

    try:
        with pytest.raises(MissingSecretsError) as info:
            run_models(
                {"first": "hi", "m": '{{ config(model_type="modal_test") }}\n{{ ref("first") }}'},
                llm_call=lambda prompt: calls.append(prompt) or "resp",
                secrets={"MODAL_TOKEN_ID": "id"},
            )
        assert "MODAL_TOKEN_SECRET" in str(info.value)
        assert "MODAL_TOKEN_ID" not in str(info.value)
        assert calls == []  # nothing ran, not even the upstream model
    finally:
        _REGISTRY.pop("modal_test", None)


# ---------------------------------------------------------------------------
# CLI: names, never values
# ---------------------------------------------------------------------------

def test_secret_flag_takes_a_literal_value():
    assert dict(parse_secret_flags([f"API_KEY={KEY}"], {})) == {"API_KEY": KEY}


def test_bare_secret_flag_reads_env_var_of_that_name():
    assert dict(parse_secret_flags(["API_KEY"], {"API_KEY": KEY})) == {"API_KEY": KEY}


def test_bare_secret_flag_rejects_unset_env_var():
    with pytest.raises(ValueError, match="API_KEY is not set"):
        parse_secret_flags(["API_KEY"], {})


def test_cli_run_exits_when_secret_env_var_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PROMPTMAP_TEST_UNSET_KEY", raising=False)
    result = CliRunner().invoke(main, ["run", "--secret", "PROMPTMAP_TEST_UNSET_KEY"])
    assert result.exit_code == 1
    assert "PROMPTMAP_TEST_UNSET_KEY is not set" in " ".join(result.output.split())


# ---------------------------------------------------------------------------
# promptmap test: --secret reaches the per-case runs and the LLM judge
# ---------------------------------------------------------------------------

SECRET_CLIENT_PY = """\
import promptmap

@promptmap.model_kind("needs_secret_cli_test", secrets={"PROMPTMAP_CLI_TEST_KEY"})
async def needs_secret(rendered, call):
    return "value-from-kind"

def llm_call(prompt, secrets):
    assert secrets["PROMPTMAP_CLI_TEST_KEY"]
    return '{"results": "pass"}'
"""


def _secret_project(root):
    (root / "map").mkdir()
    (root / "tests").mkdir()
    (root / "client.py").write_text(SECRET_CLIENT_PY)
    (root / "map" / "m.prompt").write_text(
        '{{ config(model_type="needs_secret_cli_test") }}\nhi'
    )
    (root / "tests" / "t.prompt").write_text("Is {{ ref('m') }} fine?")
    (root / "tests" / "cases.yml").write_text("cases:\n  - name: one\n")


def test_promptmap_test_passes_secrets_to_runs_and_judge(tmp_path, monkeypatch):
    _secret_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["test", "--secret", f"PROMPTMAP_CLI_TEST_KEY={KEY}"])
    assert result.exit_code == 0, result.output
    assert KEY not in result.output
    _REGISTRY.pop("needs_secret_cli_test", None)


def test_promptmap_test_without_secret_fails_cleanly(tmp_path, monkeypatch):
    _secret_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["test"])
    assert result.exit_code == 1
    assert "PROMPTMAP_CLI_TEST_KEY" in result.output
    assert not isinstance(result.exception, MissingSecretsError)  # exited, no traceback
    _REGISTRY.pop("needs_secret_cli_test", None)
