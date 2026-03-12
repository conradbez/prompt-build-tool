"""In-memory storage backend for browser and test environments."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any


class MemoryStorageBackend:
    def __init__(self) -> None:
        self._dags: dict[str, str] = {}
        self._runs: dict[str, dict[str, Any]] = {}
        self._results: dict[str, dict[str, dict[str, Any]]] = {}
        self._tests: dict[str, list[dict[str, Any]]] = {}
        self._cache: dict[str, str] = {}

    def init_db(self) -> None:
        return None

    def create_run(self, model_count: int, dag_hash: str, git_sha: str | None = None) -> str:
        run_id = str(uuid.uuid4())
        now = _now()
        self._runs[run_id] = {
            "run_id": run_id,
            "run_date": _today(now),
            "created_at": now,
            "status": "running",
            "completed_at": None,
            "model_count": model_count,
            "git_sha": git_sha,
            "dag_hash": dag_hash,
        }
        self._results[run_id] = {}
        self._tests[run_id] = []
        return run_id

    def finish_run(self, run_id: str, status: str) -> None:
        self._runs[run_id]["status"] = status
        self._runs[run_id]["completed_at"] = _now()

    def get_latest_run_with_dag_hash(self, dag_hash: str) -> dict[str, Any] | None:
        matches = [
            row for row in self._runs.values()
            if row["dag_hash"] == dag_hash and row["status"] in {"success", "partial"}
        ]
        if not matches:
            return None
        return max(matches, key=lambda row: row["created_at"])

    def record_test_result(self, run_id: str, result: Any) -> None:
        self._tests[run_id].append(
            {
                "run_id": run_id,
                "test_name": result.test_name,
                "status": result.status,
                "prompt_rendered": result.prompt_rendered,
                "llm_output": result.llm_output,
                "error": result.error or None,
                "completed_at": _now(),
                "execution_ms": result.execution_ms,
            }
        )

    def get_test_results(self, run_id: str) -> list[dict[str, Any]]:
        return list(self._tests.get(run_id, []))

    def get_model_outputs_from_run(self, run_id: str, model_names: list[str]) -> dict[str, str]:
        results = self._results.get(run_id, {})
        return {
            name: row["llm_output"]
            for name, row in results.items()
            if name in model_names and row["status"] == "success"
        }

    def save_dag(self, dag_hash: str, dag_json: str) -> None:
        self._dags.setdefault(dag_hash, dag_json)

    def load_dag(self, dag_hash: str) -> str | None:
        return self._dags.get(dag_hash)

    def get_cached_llm_output(self, cache_key: str) -> str | None:
        return self._cache.get(_prompt_hash(cache_key))

    def upsert_model_pending(
        self,
        run_id: str,
        model_name: str,
        prompt_template: str,
        depends_on: list[str],
    ) -> None:
        self._results[run_id][model_name] = {
            "run_id": run_id,
            "model_name": model_name,
            "status": "pending",
            "prompt_template": prompt_template,
            "prompt_rendered": None,
            "prompt_hash": None,
            "llm_output": None,
            "started_at": None,
            "completed_at": None,
            "execution_ms": None,
            "error": None,
            "depends_on": json.dumps(depends_on),
        }

    def mark_model_running(self, run_id: str, model_name: str) -> None:
        self._results[run_id][model_name]["status"] = "running"
        self._results[run_id][model_name]["started_at"] = _now()

    def mark_model_success(
        self,
        run_id: str,
        model_name: str,
        prompt_rendered: str,
        llm_output: str,
        cache_key: str | None = None,
    ) -> None:
        row = self._results[run_id][model_name]
        now = _now()
        started_at = row.get("started_at")
        elapsed = 0
        if started_at:
            try:
                elapsed = int(
                    (datetime.fromisoformat(now) - datetime.fromisoformat(started_at)).total_seconds() * 1000
                )
            except ValueError:
                elapsed = 0

        row.update(
            {
                "status": "success",
                "prompt_rendered": prompt_rendered,
                "prompt_hash": _prompt_hash(cache_key or prompt_rendered),
                "llm_output": llm_output,
                "completed_at": now,
                "execution_ms": elapsed,
            }
        )
        self._cache[_prompt_hash(cache_key or prompt_rendered)] = llm_output

    def mark_model_error(self, run_id: str, model_name: str, error: str) -> None:
        self._results[run_id][model_name].update(
            {
                "status": "error",
                "completed_at": _now(),
                "error": error,
            }
        )

    def mark_model_skipped(self, run_id: str, model_name: str) -> None:
        self._results[run_id][model_name]["status"] = "skipped"

    def get_run_results(self, run_id: str) -> list[dict[str, Any]]:
        return list(self._results.get(run_id, {}).values())

    def get_latest_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        return sorted(
            self._runs.values(),
            key=lambda row: row["created_at"],
            reverse=True,
        )[:limit]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _today(now: str) -> str:
    return now[:10]


def _prompt_hash(cache_key: str) -> str:
    return hashlib.sha256(cache_key.encode()).hexdigest()
