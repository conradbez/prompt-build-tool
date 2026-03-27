"""Default SQLite storage accessors used by the CLI and docs code."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pbt.storage.sqlite import SQLiteStorageBackend

_DEFAULT_BACKEND = SQLiteStorageBackend()


def db_path() -> Path:
    return _DEFAULT_BACKEND.db_path()


def get_conn():
    return _DEFAULT_BACKEND.get_conn()


def init_db() -> None:
    _DEFAULT_BACKEND.init_db()


def create_run(model_count: int, git_sha: Optional[str] = None) -> str:
    return _DEFAULT_BACKEND.create_run(model_count=model_count, git_sha=git_sha)


def finish_run(run_id: str, status: str) -> None:
    _DEFAULT_BACKEND.finish_run(run_id, status)


def get_latest_successful_run():
    return _DEFAULT_BACKEND.get_latest_successful_run()


def record_test_result(run_id: str, result: "TestResult") -> None:  # noqa: F821
    _DEFAULT_BACKEND.record_test_result(run_id, result)


def get_test_results(run_id: str):
    return _DEFAULT_BACKEND.get_test_results(run_id)


def get_model_outputs_from_run(run_id: str, model_names: list[str]) -> dict[str, str]:
    return _DEFAULT_BACKEND.get_model_outputs_from_run(run_id, model_names)


def get_cached_llm_output(cache_key: str) -> str | None:
    return _DEFAULT_BACKEND.get_cached_llm_output(cache_key)


def upsert_model_pending(run_id: str, model_name: str, prompt_template: str, depends_on: list[str]) -> None:
    _DEFAULT_BACKEND.upsert_model_pending(run_id, model_name, prompt_template, depends_on)


def mark_model_running(run_id: str, model_name: str) -> None:
    _DEFAULT_BACKEND.mark_model_running(run_id, model_name)


def mark_model_success(
    run_id: str,
    model_name: str,
    prompt_rendered: str,
    llm_output: str,
    cache_key: str | None = None,
) -> None:
    _DEFAULT_BACKEND.mark_model_success(run_id, model_name, prompt_rendered, llm_output, cache_key=cache_key)


def mark_model_error(run_id: str, model_name: str, error: str) -> None:
    _DEFAULT_BACKEND.mark_model_error(run_id, model_name, error)


def mark_model_skipped(run_id: str, model_name: str) -> None:
    _DEFAULT_BACKEND.mark_model_skipped(run_id, model_name)


def get_run_results(run_id: str):
    return _DEFAULT_BACKEND.get_run_results(run_id)


def get_latest_runs(limit: int = 10):
    return _DEFAULT_BACKEND.get_latest_runs(limit)


def clear_cache() -> int:
    return _DEFAULT_BACKEND.clear_cache()
