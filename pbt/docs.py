"""
pbt docs — generate a self-contained HTML report of all previous runs.

The generated HTML includes:
  - A summary table of every pbt run (status, models, timing, DAG hash)
  - Expandable per-run model results
  - A Mermaid.js DAG diagram of the current model dependency graph
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from pbt.model_spec import ModelSpec


_TEMPLATES = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent / "html_templates")),
    autoescape=True,
)

STATUS_COLOURS = {
    "success": "#22c55e",
    "error":   "#ef4444",
    "partial": "#f59e0b",
    "running": "#3b82f6",
    "pass":    "#22c55e",
    "fail":    "#ef4444",
    "skipped": "#a3a3a3",
    "pending": "#a3a3a3",
}


def _mermaid_dag(models: dict[str, "ModelSpec"]) -> str:
    """Return a Mermaid flowchart string for the model DAG."""
    lines = ["graph LR"]
    for name in sorted(models):
        safe = name.replace("-", "_")
        lines.append(f"    {safe}[{name}]")
    for name, model in sorted(models.items()):
        safe_dst = name.replace("-", "_")
        for dep in model.depends_on:
            safe_src = dep.replace("-", "_")
            lines.append(f"    {safe_src} --> {safe_dst}")
    return "\n".join(lines)


def _duration(created_at: str | None, completed_at: str | None) -> str:
    if not created_at or not completed_at:
        return "—"
    try:
        secs = int((datetime.fromisoformat(completed_at) - datetime.fromisoformat(created_at)).total_seconds())
        return f"{secs}s"
    except Exception:
        return "—"


#: Longest per-model text embedded in the report for the click-to-open modal.
_FULL_LIMIT = 20000


def generate_docs(
    runs: list,           # list of sqlite3.Row from runs table
    run_results: dict,    # run_id -> list[sqlite3.Row] from model_results
    models: dict | None,  # dict[name, PromptModel] or None if no models dir
    output_path: Path,
) -> None:
    """Write a self-contained HTML docs file to *output_path*."""

    def _column(row, name: str):
        """Read *name* from a storage row, or None when the backend lacks it."""
        keys = row.keys() if hasattr(row, "keys") else row
        return row[name] if name in keys else None

    def _type_label(row) -> str:
        """How the model's type would be written in its config() block.

        A model with no ``model_type`` is a plain LLM call; there is nothing to
        write for it, so it is shown as "prompt".
        """
        return _column(row, "model_type") or "prompt"

    def _config_summary(row) -> str:
        """The model's config() options, minus the type shown in its own column."""
        try:
            config = json.loads(_column(row, "config") or "{}")
        except (TypeError, ValueError):
            return ""
        config.pop("model_type", None)
        return ", ".join(f"{key}={value}" for key, value in sorted(config.items()))

    def _model_output(row) -> str:
        """The output this model actually passed downstream.

        ``llm_output`` holds the raw response, which is what the prompt cache
        serves; a validator or a post-processing model kind may have changed it
        before anything downstream saw it.
        """
        return _column(row, "llm_output_validated") or row["llm_output"]

    runs_data = []
    for run in runs:
        rid = run["run_id"]
        results = [
            {
                "model_name": r["model_name"],
                "model_type": _type_label(r),
                "config": _config_summary(r),
                "status": r["status"] or "—",
                "execution_ms": r["execution_ms"],
                "error": r["error"] or "",
                # Previews fill the table; the full text is carried alongside so
                # clicking a cell can open it without another request.
                "input": (r["prompt_rendered"] or "")[:_FULL_LIMIT],
                "input_preview": (r["prompt_rendered"] or "")[:200],
                "output": (_model_output(r) or "")[:_FULL_LIMIT],
                "output_preview": (_model_output(r) or "")[:200],
                "cached": bool(_column(r, "cached")),
            }
            for r in run_results.get(rid, [])
        ]
        runs_data.append({
            "run_id": rid,
            "short_id": rid[:8] + "…",
            "status": run["status"] or "—",
            "model_count": run["model_count"] or 0,
            "created": (run["created_at"] or "")[:19].replace("T", " "),
            "duration": _duration(run["created_at"], run["completed_at"]),
            "results": results,
        })

    html = _TEMPLATES.get_template("docs.html").render(
        runs=runs_data,
        status_colours=STATUS_COLOURS,
        dag_section=_mermaid_dag(models) if models else "",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
