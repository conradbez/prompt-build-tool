"""SQLite-backed storage backend."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class SQLiteStorageBackend:
    def __init__(self, path: str | Path | None = None) -> None:
        self._db_path = Path(path) if path is not None else Path(".pbt") / "pbt.db"

    def db_path(self) -> Path:
        return self._db_path

    @contextmanager
    def get_conn(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id       TEXT        PRIMARY KEY,
                    run_date     TEXT        NOT NULL,
                    created_at   TEXT        NOT NULL,
                    status       TEXT        NOT NULL DEFAULT 'running',
                    completed_at TEXT,
                    model_count  INTEGER     NOT NULL DEFAULT 0,
                    git_sha      TEXT
                );

                CREATE TABLE IF NOT EXISTS model_results (
                    id               INTEGER   PRIMARY KEY AUTOINCREMENT,
                    run_id           TEXT      NOT NULL REFERENCES runs(run_id),
                    model_name       TEXT      NOT NULL,
                    status           TEXT      NOT NULL DEFAULT 'pending',
                    prompt_template  TEXT,
                    prompt_rendered  TEXT,
                    prompt_hash      TEXT,
                    llm_output       TEXT,
                    started_at       TEXT,
                    completed_at     TEXT,
                    execution_ms     INTEGER,
                    error            TEXT,
                    depends_on       TEXT      NOT NULL DEFAULT '[]'
                );

                CREATE INDEX IF NOT EXISTS idx_model_results_run
                    ON model_results (run_id, model_name);

                CREATE INDEX IF NOT EXISTS idx_model_results_prompt_hash
                    ON model_results (prompt_hash, completed_at DESC);

                CREATE TABLE IF NOT EXISTS test_results (
                    id               INTEGER   PRIMARY KEY AUTOINCREMENT,
                    run_id           TEXT      NOT NULL REFERENCES runs(run_id),
                    test_name        TEXT      NOT NULL,
                    status           TEXT      NOT NULL,
                    prompt_rendered  TEXT,
                    llm_output       TEXT,
                    error            TEXT,
                    started_at       TEXT,
                    completed_at     TEXT,
                    execution_ms     INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_test_results_run
                    ON test_results (run_id, test_name);
            """)

    def create_run(self, model_count: int, git_sha: Optional[str] = None) -> str:
        run_id = str(uuid.uuid4())
        now = _now()
        with self.get_conn() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, run_date, created_at, status, model_count, git_sha) "
                "VALUES (?, ?, ?, 'running', ?, ?)",
                (run_id, _today(now), now, model_count, git_sha),
            )
        return run_id

    def finish_run(self, run_id: str, status: str) -> None:
        with self.get_conn() as conn:
            conn.execute(
                "UPDATE runs SET status=?, completed_at=? WHERE run_id=?",
                (status, _now(), run_id),
            )

    def get_latest_successful_run(self) -> Optional[sqlite3.Row]:
        with self.get_conn() as conn:
            return conn.execute(
                """SELECT * FROM runs
                   WHERE status IN ('success', 'partial')
                   ORDER BY created_at DESC
                   LIMIT 1""",
            ).fetchone()

    def record_test_result(self, run_id: str, result: "TestResult") -> None:  # noqa: F821
        with self.get_conn() as conn:
            conn.execute(
                """INSERT INTO test_results
                   (run_id, test_name, status, prompt_rendered, llm_output,
                    error, completed_at, execution_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    result.test_name,
                    result.status,
                    result.prompt_rendered,
                    result.llm_output,
                    result.error or None,
                    _now(),
                    result.execution_ms,
                ),
            )

    def get_test_results(self, run_id: str) -> list[sqlite3.Row]:
        with self.get_conn() as conn:
            return conn.execute(
                "SELECT * FROM test_results WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()

    def get_model_outputs_from_run(self, run_id: str, model_names: list[str]) -> dict[str, str]:
        if not model_names:
            return {}
        placeholders = ",".join("?" * len(model_names))
        with self.get_conn() as conn:
            rows = conn.execute(
                f"""SELECT model_name, llm_output FROM model_results
                    WHERE run_id = ?
                      AND model_name IN ({placeholders})
                      AND status = 'success'""",
                (run_id, *model_names),
            ).fetchall()
        return {row["model_name"]: row["llm_output"] for row in rows}

    def get_cached_llm_output(self, cache_key: str) -> Optional[str]:
        prompt_hash = hashlib.sha256(cache_key.encode()).hexdigest()
        with self.get_conn() as conn:
            row = conn.execute(
                """SELECT llm_output FROM model_results
                   WHERE prompt_hash = ? AND status = 'success'
                   ORDER BY completed_at DESC
                   LIMIT 1""",
                (prompt_hash,),
            ).fetchone()
        return row["llm_output"] if row else None

    def upsert_model_pending(
        self,
        run_id: str,
        model_name: str,
        prompt_template: str,
        depends_on: list[str],
    ) -> None:
        with self.get_conn() as conn:
            conn.execute(
                """INSERT INTO model_results
                   (run_id, model_name, status, prompt_template, depends_on)
                   VALUES (?, ?, 'pending', ?, ?)
                """,
                (run_id, model_name, prompt_template, json.dumps(depends_on)),
            )

    def mark_model_running(self, run_id: str, model_name: str) -> None:
        with self.get_conn() as conn:
            conn.execute(
                "UPDATE model_results SET status='running', started_at=? "
                "WHERE run_id=? AND model_name=?",
                (_now(), run_id, model_name),
            )

    def mark_model_success(
        self,
        run_id: str,
        model_name: str,
        prompt_rendered: str,
        llm_output: str,
        cache_key: str | None = None,
    ) -> None:
        now = _now()
        with self.get_conn() as conn:
            row = conn.execute(
                "SELECT started_at FROM model_results WHERE run_id=? AND model_name=?",
                (run_id, model_name),
            ).fetchone()
            started_str = row["started_at"] if row else None
            if started_str:
                try:
                    started_dt = datetime.fromisoformat(started_str)
                    now_dt = datetime.fromisoformat(now)
                    elapsed = int((now_dt - started_dt).total_seconds() * 1000)
                except (ValueError, TypeError):
                    elapsed = 0
            else:
                elapsed = 0

            prompt_hash = hashlib.sha256((cache_key or prompt_rendered).encode()).hexdigest()
            conn.execute(
                """UPDATE model_results
                   SET status='success',
                       prompt_rendered=?,
                       prompt_hash=?,
                       llm_output=?,
                       completed_at=?,
                       execution_ms=?
                   WHERE run_id=? AND model_name=?
                """,
                (prompt_rendered, prompt_hash, llm_output, now, elapsed, run_id, model_name),
            )

    def mark_model_error(self, run_id: str, model_name: str, error: str) -> None:
        with self.get_conn() as conn:
            conn.execute(
                """UPDATE model_results
                   SET status='error', completed_at=?, error=?
                   WHERE run_id=? AND model_name=?
                """,
                (_now(), error, run_id, model_name),
            )

    def mark_model_skipped(self, run_id: str, model_name: str) -> None:
        with self.get_conn() as conn:
            conn.execute(
                "UPDATE model_results SET status='skipped' "
                "WHERE run_id=? AND model_name=?",
                (run_id, model_name),
            )

    def get_run_results(self, run_id: str) -> list[sqlite3.Row]:
        with self.get_conn() as conn:
            return conn.execute(
                "SELECT * FROM model_results WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()

    def get_latest_runs(self, limit: int = 10) -> list[sqlite3.Row]:
        with self.get_conn() as conn:
            return conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def clear_cache(self) -> int:
        """Nullify all prompt_hash values, forcing fresh LLM calls on next run.

        Returns the number of cache entries cleared.
        """
        with self.get_conn() as conn:
            cursor = conn.execute(
                "UPDATE model_results SET prompt_hash = NULL WHERE prompt_hash IS NOT NULL"
            )
            return cursor.rowcount


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _today(now: str) -> str:
    return now[:10]
