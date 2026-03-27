"""
pbt docs — generate a self-contained HTML report of all previous runs.

The generated HTML includes:
  - A summary table of every pbt run (status, models, timing, DAG hash)
  - Expandable per-run model results
  - A Mermaid.js DAG diagram of the current model dependency graph
"""

from __future__ import annotations

import html as _html
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from pbt.executor.model_constructs import BaseModelHandler


_TEMPLATES = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent / "html_templates")),
    autoescape=True,
)

_STATUS_COLOUR = {
    "success": "#22c55e",
    "error":   "#ef4444",
    "partial": "#f59e0b",
    "running": "#3b82f6",
    "pass":    "#22c55e",
    "fail":    "#ef4444",
    "skipped": "#a3a3a3",
    "pending": "#a3a3a3",
}


def _esc(s: str | None) -> str:
    return _html.escape(str(s or ""), quote=True)


def _badge(status: str) -> str:
    colour = _STATUS_COLOUR.get(status, "#6b7280")
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.8em;font-weight:600;">'
        f"{_esc(status)}</span>"
    )


def _mermaid_dag(models: dict[str, "BaseModelHandler"]) -> str:
    """Return a Mermaid flowchart string for the model DAG."""
    lines = ["graph LR"]
    for name in sorted(models):
        safe = name.replace("-", "_")
        lines.append(f"    {safe}[{_esc(name)}]")
    for name, model in sorted(models.items()):
        safe_dst = name.replace("-", "_")
        for dep in model.depends_on:
            safe_src = dep.replace("-", "_")
            lines.append(f"    {safe_src} --> {safe_dst}")
    return "\n".join(lines)


def generate_docs(
    runs: list,           # list of sqlite3.Row from runs table
    run_results: dict,    # run_id -> list[sqlite3.Row] from model_results
    models: dict | None,  # dict[name, PromptModel] or None if no models dir
    output_path: Path,
) -> None:
    """Write a self-contained HTML docs file to *output_path*."""

    # -----------------------------------------------------------------------
    # Runs table rows HTML
    # -----------------------------------------------------------------------
    runs_rows = []
    for run in runs:
        rid = run["run_id"]
        short_id = rid[:8] + "…"
        status = run["status"] or "—"
        date = run["run_date"] or "—"
        models_count = run["model_count"] or 0
        created = (run["created_at"] or "")[:19].replace("T", " ")
        duration = "—"
        if run["created_at"] and run["completed_at"]:
            try:
                t0 = datetime.fromisoformat(run["created_at"])
                t1 = datetime.fromisoformat(run["completed_at"])
                secs = int((t1 - t0).total_seconds())
                duration = f"{secs}s"
            except Exception:
                pass

        results = run_results.get(rid, [])
        results_html = ""
        if results:
            result_rows = []
            for r in results:
                ms = r["execution_ms"] or 0
                ms_str = f"{ms} ms" if ms else "—"
                err = _esc(r["error"]) if r["error"] else ""
                output_preview = _esc((r["llm_output"] or "")[:200])
                result_rows.append(
                    f"<tr>"
                    f"<td style='font-family:monospace'>{_esc(r['model_name'])}</td>"
                    f"<td>{_badge(r['status'])}</td>"
                    f"<td style='color:var(--pico-muted-color)'>{ms_str}</td>"
                    f"<td style='font-size:0.85em;max-width:400px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis'>"
                    f"{err or output_preview}</td>"
                    f"</tr>"
                )
            results_html = (
                f"<tr id='detail-{_esc(rid)}' style='display:none'>"
                f"<td colspan='6' style='padding:0'>"
                f"<div style='padding:var(--pico-spacing);background:var(--pico-card-sectioning-background-color)'>"
                f"<table style='margin-bottom:0'>"
                f"<thead><tr><th>Model</th><th>Status</th><th>Time</th><th>Output preview</th></tr></thead>"
                f"<tbody>{''.join(result_rows)}</tbody></table></div></td></tr>"
            )

        toggle_js = f"toggleDetail('{_esc(rid)}')"
        runs_rows.append(
            f"<tr onclick=\"{toggle_js}\" style='cursor:pointer'>"
            f"<td style='font-family:monospace;font-size:0.85em'>{_esc(short_id)}</td>"
            f"<td>{date}</td>"
            f"<td>{_badge(status)}</td>"
            f"<td style='text-align:right'>{models_count}</td>"
            f"<td style='color:var(--pico-muted-color);font-size:0.9em'>{created}</td>"
            f"<td style='color:var(--pico-muted-color);font-size:0.9em'>{duration}</td>"
            f"</tr>"
            + results_html
        )

    runs_table_body = "\n".join(runs_rows) if runs_rows else (
        "<tr><td colspan='6' style='text-align:center;color:var(--pico-muted-color)'>"
        "No runs recorded yet. Run <code>pbt run</code> first.</td></tr>"
    )

    dag_section = _mermaid_dag(models) if models else ""

    html = _TEMPLATES.get_template("docs.html").render(
        runs_table_body=runs_table_body,
        dag_section=dag_section,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
