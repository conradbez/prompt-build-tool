"""pbt test caches each verdict, on every storage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from pbt.storage import MemoryStorageBackend
from pbt.storage.sqlite import SQLiteStorageBackend
from pbt.tester import execute_tests


@pytest.mark.parametrize("backend", ["sqlite", "memory"])
def test_identical_test_is_judged_once(tmp_path: Path, backend: str) -> None:
    storage = SQLiteStorageBackend(tmp_path / "pbt.db") if backend == "sqlite" else MemoryStorageBackend()
    storage.init_db()
    calls: list[str] = []

    def judge(prompt: str) -> str:
        calls.append(prompt)
        return '{"results": "pass"}'

    for _ in range(2):
        run_id = storage.create_run(model_count=1)
        (result,) = execute_tests(
            run_id=run_id,
            tests={"is_short": "Is this short? {{ ref('haiku') }}"},
            model_outputs={"haiku": "old pond"},
            storage_backend=storage,
            llm_call=judge,
        )
        assert result.status == "pass"
    assert len(calls) == 1


def test_cached_test_rows_are_not_listed_as_models(tmp_path: Path) -> None:
    from pbt.docs import generate_docs

    storage = SQLiteStorageBackend(tmp_path / "pbt.db")
    storage.init_db()
    run_id = storage.create_run(model_count=1)
    execute_tests(
        run_id=run_id, tests={"verdict_row_zz": "Q {{ ref('m') }}"}, model_outputs={"m": "x"},
        storage_backend=storage, llm_call=lambda p: '{"results": "pass"}',
    )
    out = tmp_path / "docs" / "index.html"
    generate_docs(
        runs=list(storage.get_latest_runs()),
        run_results={run_id: storage.get_run_results(run_id)},
        models=None, output_path=out, blob_store=storage.blob_store(),
    )
    html = out.read_text()
    assert "verdict_row_zz" not in html
