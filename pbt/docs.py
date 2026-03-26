"""
pbt docs — generate a self-contained HTML report of all previous runs.

The generated HTML includes:
  - A summary table of every pbt run (status, models, timing, DAG hash)
  - Expandable per-run model results
  - A Mermaid.js DAG diagram of the current model dependency graph
"""

from __future__ import annotations

import json
import html as _html
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pbt.executor.model_constructs import BaseModelHandler


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
    for name, model in sorted(models.items()):
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
        completed = (run["completed_at"] or "")[:19].replace("T", " ") or "—"
        duration = "—"
        if run["created_at"] and run["completed_at"]:
            try:
                from datetime import datetime
                t0 = datetime.fromisoformat(run["created_at"])
                t1 = datetime.fromisoformat(run["completed_at"])
                secs = int((t1 - t0).total_seconds())
                duration = f"{secs}s"
            except Exception:
                pass

        # Model results sub-rows
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
                    f"<td style='padding:4px 12px;font-family:monospace'>{_esc(r['model_name'])}</td>"
                    f"<td style='padding:4px 12px'>{_badge(r['status'])}</td>"
                    f"<td style='padding:4px 12px;color:#6b7280'>{ms_str}</td>"
                    f"<td style='padding:4px 12px;font-size:0.85em;color:#374151;max-width:400px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis'>"
                    f"{err or output_preview}</td>"
                    f"</tr>"
                )
            results_html = (
                f"<tr id='detail-{_esc(rid)}' style='display:none'>"
                f"<td colspan='6' style='padding:0 16px 16px 16px;background:#f9fafb'>"
                f"<table style='width:100%;border-collapse:collapse;font-size:0.9em'>"
                f"<thead><tr style='text-align:left;color:#6b7280'>"
                f"<th style='padding:4px 12px'>Model</th>"
                f"<th style='padding:4px 12px'>Status</th>"
                f"<th style='padding:4px 12px'>Time</th>"
                f"<th style='padding:4px 12px'>Output preview</th>"
                f"</tr></thead><tbody>"
                + "".join(result_rows)
                + "</tbody></table></td></tr>"
            )

        toggle_js = f"toggleDetail('{_esc(rid)}')"
        runs_rows.append(
            f"<tr onclick=\"{toggle_js}\" style='cursor:pointer;border-bottom:1px solid #e5e7eb'>"
            f"<td style='padding:10px 16px;font-family:monospace;font-size:0.85em'>{_esc(short_id)}</td>"
            f"<td style='padding:10px 16px'>{date}</td>"
            f"<td style='padding:10px 16px'>{_badge(status)}</td>"
            f"<td style='padding:10px 16px;text-align:right'>{models_count}</td>"
            f"<td style='padding:10px 16px;color:#6b7280;font-size:0.9em'>{created}</td>"
            f"<td style='padding:10px 16px;color:#6b7280;font-size:0.9em'>{duration}</td>"
            f"</tr>"
            + results_html
        )

    runs_table_body = "\n".join(runs_rows) if runs_rows else (
        "<tr><td colspan='6' style='padding:24px;text-align:center;color:#9ca3af'>"
        "No runs recorded yet. Run <code>pbt run</code> first.</td></tr>"
    )

    # -----------------------------------------------------------------------
    # DAG section
    # -----------------------------------------------------------------------
    if models:
        dag_mermaid = _mermaid_dag(models)
        dag_section = f"""
<section style="margin-top:40px">
  <h2 style="font-size:1.2em;font-weight:600;margin-bottom:12px">Model DAG</h2>
  <div class="mermaid" style="background:#f9fafb;padding:24px;border-radius:8px;overflow:auto">
{_esc(dag_mermaid)}
  </div>
</section>
"""
    else:
        dag_section = ""

    # -----------------------------------------------------------------------
    # Full HTML
    # -----------------------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>pbt docs</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 0; padding: 32px; background: #fff; color: #111; }}
    h1   {{ font-size: 1.6em; font-weight: 700; margin-bottom: 4px; }}
    .subtitle {{ color: #6b7280; margin-bottom: 32px; font-size: 0.95em; }}
    table {{ width: 100%; border-collapse: collapse; }}
    thead tr {{ background: #f3f4f6; text-align: left; }}
    th {{ padding: 10px 16px; font-size: 0.85em; font-weight: 600;
          text-transform: uppercase; letter-spacing: 0.05em; color: #374151; }}
    tr:hover {{ background: #f9fafb; }}
    code {{ background: #f3f4f6; padding: 1px 5px; border-radius: 3px;
            font-family: monospace; font-size: 0.9em; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }}
  </style>
</head>
<body>
  <h1>pbt docs</h1>
  <p class="subtitle">prompt-build-tool run history &mdash; generated {_esc(__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M'))}</p>

  <section>
    <h2 style="font-size:1.2em;font-weight:600;margin-bottom:12px">Run History</h2>
    <p style="font-size:0.88em;color:#6b7280;margin-bottom:8px">Click a row to expand model details.</p>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Date</th>
            <th>Status</th>
            <th style="text-align:right">Models</th>
            <th>Started</th>
            <th>Duration</th>
          </tr>
        </thead>
        <tbody>
{runs_table_body}
        </tbody>
      </table>
    </div>
  </section>

{dag_section}

  <script>
    mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});

    function toggleDetail(runId) {{
      var row = document.getElementById('detail-' + runId);
      if (row) {{
        row.style.display = row.style.display === 'none' ? 'table-row' : 'none';
      }}
    }}
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
