"""
SQLite storage layer — schema inspired by dbt's run artifacts.

Tables
------
runs
    One row per `pbt run` invocation.

model_results
    One row per model per run — stores rendered prompt input and LLM output.
"""

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
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
    """Create tables if they don't exist, then apply any pending migrations."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id       TEXT        PRIMARY KEY,
                run_date     TEXT        NOT NULL,   -- YYYY-MM-DD, for easy date-based filtering
                created_at   TEXT        NOT NULL,   -- ISO-8601 UTC datetime
                status       TEXT        NOT NULL DEFAULT 'running',
                              -- 'running' | 'success' | 'error' | 'partial'
                completed_at TEXT,
                model_count  INTEGER     NOT NULL DEFAULT 0,
                git_sha      TEXT,
                dag_hash     TEXT        -- short SHA256 of the DAG structure
            );

            CREATE TABLE IF NOT EXISTS model_results (
                id               INTEGER   PRIMARY KEY AUTOINCREMENT,
                run_id           TEXT      NOT NULL REFERENCES runs(run_id),
                model_name       TEXT      NOT NULL,
                status           TEXT      NOT NULL DEFAULT 'pending',
                                  -- 'pending' | 'running' | 'success' | 'error' | 'skipped'
                prompt_template  TEXT,     -- raw .prompt file contents
                prompt_rendered  TEXT,     -- fully-rendered prompt sent to LLM
                prompt_hash      TEXT,     -- SHA-256 of prompt_rendered (for cache lookup)
                llm_output       TEXT,     -- raw LLM response text
                started_at       TEXT,
                completed_at     TEXT,
                execution_ms     INTEGER,
                error            TEXT,
                depends_on       TEXT      NOT NULL DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_model_results_run
                ON model_results (run_id, model_name);

            CREATE INDEX IF NOT EXISTS idx_runs_dag_hash
                ON runs (dag_hash, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_model_results_prompt_hash
                ON model_results (prompt_hash, completed_at DESC);

            CREATE TABLE IF NOT EXISTS dags (
                dag_hash   TEXT PRIMARY KEY,
                dag_json   TEXT NOT NULL,  -- JSON-serialised models snapshot
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS test_results (
                id               INTEGER   PRIMARY KEY AUTOINCREMENT,
                run_id           TEXT      NOT NULL REFERENCES runs(run_id),
                test_name        TEXT      NOT NULL,
                status           TEXT      NOT NULL,
                                  -- 'pass' | 'fail' | 'error'
                prompt_rendered  TEXT,     -- fully-rendered prompt sent to LLM
                llm_output       TEXT,     -- raw LLM response text
                error            TEXT,
                started_at       TEXT,
                completed_at     TEXT,
                execution_ms     INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_test_results_run
                ON test_results (run_id, test_name);
        """)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """
    Idempotent migrations for databases created before new columns existed.
    ALTER TABLE ADD COLUMN is a no-op if the column is already present.
    """
    migrations = [
        ("runs", "run_date",  "TEXT NOT NULL DEFAULT ''"),
        ("runs", "dag_hash",  "TEXT"),
        ("model_results", "prompt_hash", "TEXT"),
    ]
    for table, col, defn in migrations:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass  # column already exists


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def create_run(
    model_count: int,
    dag_hash: str,
    git_sha: Optional[str] = None,
) -> str:
    run_id = str(uuid.uuid4())
    now = _now()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, run_date, created_at, status, model_count, git_sha, dag_hash) "
            "VALUES (?, ?, ?, 'running', ?, ?, ?)",
            (run_id, _today(now), now, model_count, git_sha, dag_hash),
        )
    return run_id


def finish_run(run_id: str, status: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE runs SET status=?, completed_at=? WHERE run_id=?",
            (status, _now(), run_id),
        )


# ---------------------------------------------------------------------------
# Previous-run lookup (for --select)
# ---------------------------------------------------------------------------

def get_latest_run_with_dag_hash(dag_hash: str) -> Optional[sqlite3.Row]:
    """
    Return the most recent successful (or partial) run that matches *dag_hash*.
    Returns None if no such run exists.
    """
    with get_conn() as conn:
        return conn.execute(
            """SELECT * FROM runs
               WHERE dag_hash = ?
                 AND status IN ('success', 'partial')
               ORDER BY created_at DESC
               LIMIT 1""",
            (dag_hash,),
        ).fetchone()


def record_test_result(run_id: str, result: "TestResult") -> None:  # noqa: F821
    """Persist a single test outcome to the test_results table."""
    with get_conn() as conn:
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


def get_test_results(run_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM test_results WHERE run_id=? ORDER BY id",
            (run_id,),
        ).fetchall()


def get_model_outputs_from_run(
    run_id: str,
    model_names: list[str],
) -> dict[str, str]:
    """
    Fetch the LLM outputs for *model_names* from a specific run.
    Only returns models that completed with status='success'.
    """
    if not model_names:
        return {}
    placeholders = ",".join("?" * len(model_names))
    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT model_name, llm_output FROM model_results
                WHERE run_id = ?
                  AND model_name IN ({placeholders})
                  AND status = 'success'""",
            (run_id, *model_names),
        ).fetchall()
    return {row["model_name"]: row["llm_output"] for row in rows}


# ---------------------------------------------------------------------------
# DAG snapshots
# ---------------------------------------------------------------------------

def save_dag(dag_hash: str, dag_json: str) -> None:
    """
    Persist a DAG snapshot (serialised models) keyed by *dag_hash*.
    Uses INSERT OR IGNORE so repeated calls for the same hash are no-ops.
    """
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO dags (dag_hash, dag_json, created_at) VALUES (?, ?, ?)",
            (dag_hash, dag_json, _now()),
        )


def load_dag(dag_hash: str) -> Optional[str]:
    """
    Return the JSON string for a previously saved DAG snapshot, or None.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT dag_json FROM dags WHERE dag_hash = ?",
            (dag_hash,),
        ).fetchone()
    return row["dag_json"] if row else None


# ---------------------------------------------------------------------------
# Prompt cache
# ---------------------------------------------------------------------------

def get_cached_llm_output(cache_key: str) -> Optional[str]:
    """
    Return a previously stored LLM output whose cache key matches *cache_key*,
    or None if no cached result exists.

    *cache_key* is the string that gets SHA-256 hashed for the lookup.
    Callers should include all inputs that affect the LLM response
    (rendered prompt text, model config, etc.).
    """
    prompt_hash = hashlib.sha256(cache_key.encode()).hexdigest()
    with get_conn() as conn:
        row = conn.execute(
            """SELECT llm_output FROM model_results
               WHERE prompt_hash = ? AND status = 'success'
               ORDER BY completed_at DESC
               LIMIT 1""",
            (prompt_hash,),
        ).fetchone()
    return row["llm_output"] if row else None


# ---------------------------------------------------------------------------
# Model result helpers
# ---------------------------------------------------------------------------

def upsert_model_pending(
    run_id: str,
    model_name: str,
    prompt_template: str,
    depends_on: list[str],
) -> None:
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
    cache_key: str | None = None,
) -> None:
    now = _now()
    with get_conn() as conn:
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

def _now() -> str:
    """ISO-8601 UTC datetime string, stored as TEXT in SQLite."""
    return datetime.utcnow().isoformat(timespec="milliseconds")


def _today(now: str) -> str:
    """Extract YYYY-MM-DD from an ISO datetime string."""
    return now[:10]
