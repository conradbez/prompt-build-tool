"""
SQLite storage layer — schema inspired by dbt's run artifacts.

Tables
------
runs
    One row per `pbt run` invocation.

model_results
    One row per model per run — stores rendered prompt input and LLM output.
"""

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DB_PATH = Path(".pbt") / "pbt.db"


def db_path() -> Path:
    return _DB_PATH


@contextmanager
def get_conn():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
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


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id       TEXT        PRIMARY KEY,
                created_at   TIMESTAMP   NOT NULL,
                status       TEXT        NOT NULL DEFAULT 'running',
                              -- 'running' | 'success' | 'error' | 'partial'
                completed_at TIMESTAMP,
                model_count  INTEGER     NOT NULL DEFAULT 0,
                git_sha      TEXT
            );

            CREATE TABLE IF NOT EXISTS model_results (
                id               INTEGER   PRIMARY KEY AUTOINCREMENT,
                run_id           TEXT      NOT NULL REFERENCES runs(run_id),
                model_name       TEXT      NOT NULL,
                status           TEXT      NOT NULL DEFAULT 'pending',
                                  -- 'pending' | 'running' | 'success' | 'error' | 'skipped'
                prompt_template  TEXT,     -- raw .prompt file contents
                prompt_rendered  TEXT,     -- fully-rendered prompt sent to LLM
                llm_output       TEXT,     -- raw LLM response text
                started_at       TIMESTAMP,
                completed_at     TIMESTAMP,
                execution_ms     INTEGER,
                error            TEXT,
                -- dependency chain (JSON list of model names)
                depends_on       TEXT      NOT NULL DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_model_results_run
                ON model_results (run_id, model_name);
        """)


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def create_run(model_count: int, git_sha: Optional[str] = None) -> str:
    run_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, created_at, status, model_count, git_sha) "
            "VALUES (?, ?, 'running', ?, ?)",
            (run_id, _now(), model_count, git_sha),
        )
    return run_id


def finish_run(run_id: str, status: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE runs SET status=?, completed_at=? WHERE run_id=?",
            (status, _now(), run_id),
        )


# ---------------------------------------------------------------------------
# Model result helpers
# ---------------------------------------------------------------------------

def upsert_model_pending(
    run_id: str,
    model_name: str,
    prompt_template: str,
    depends_on: list[str],
) -> None:
    import json
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO model_results
               (run_id, model_name, status, prompt_template, depends_on)
               VALUES (?, ?, 'pending', ?, ?)
            """,
            (run_id, model_name, prompt_template, json.dumps(depends_on)),
        )


def mark_model_running(run_id: str, model_name: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE model_results SET status='running', started_at=? "
            "WHERE run_id=? AND model_name=?",
            (_now(), run_id, model_name),
        )


def mark_model_success(
    run_id: str,
    model_name: str,
    prompt_rendered: str,
    llm_output: str,
) -> None:
    now = _now()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT started_at FROM model_results WHERE run_id=? AND model_name=?",
            (run_id, model_name),
        ).fetchone()
        started = row["started_at"] if row else now
        elapsed = int(
            (now - started).total_seconds() * 1000
        ) if isinstance(started, datetime) else 0

        conn.execute(
            """UPDATE model_results
               SET status='success',
                   prompt_rendered=?,
                   llm_output=?,
                   completed_at=?,
                   execution_ms=?
               WHERE run_id=? AND model_name=?
            """,
            (prompt_rendered, llm_output, now, elapsed, run_id, model_name),
        )


def mark_model_error(run_id: str, model_name: str, error: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """UPDATE model_results
               SET status='error', completed_at=?, error=?
               WHERE run_id=? AND model_name=?
            """,
            (_now(), error, run_id, model_name),
        )


def mark_model_skipped(run_id: str, model_name: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE model_results SET status='skipped' "
            "WHERE run_id=? AND model_name=?",
            (run_id, model_name),
        )


def get_run_results(run_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM model_results WHERE run_id=? ORDER BY id",
            (run_id,),
        ).fetchall()


def get_latest_runs(limit: int = 10) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)
