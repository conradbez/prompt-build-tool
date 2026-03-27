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


_PICO = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">'

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
                f"<thead><tr>"
                f"<th>Model</th>"
                f"<th>Status</th>"
                f"<th>Time</th>"
                f"<th>Output preview</th>"
                f"</tr></thead><tbody>"
                + "".join(result_rows)
                + "</tbody></table></div></td></tr>"
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

    # -----------------------------------------------------------------------
    # DAG section
    # -----------------------------------------------------------------------
    if models:
        dag_mermaid = _mermaid_dag(models)
        dag_section = f"""
<section>
  <h2>Model DAG</h2>
  <article>
    <div class="mermaid" style="overflow:auto">
{_esc(dag_mermaid)}
    </div>
  </article>
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
  {_PICO}
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <style>
    nav {{ margin-bottom: 0; }}
    section {{ margin-top: var(--pico-spacing); }}
    h2 {{ font-size: 1.2em; font-weight: 600; margin-bottom: var(--pico-spacing); }}
  </style>
</head>
<body>
  <main class="container">
    <hgroup>
      <h1>pbt docs</h1>
      <p>prompt-build-tool run history &mdash; generated {_esc(__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M'))}</p>
    </hgroup>

    <section>
      <h2>Run History</h2>
      <p><small>Click a row to expand model details.</small></p>
      <figure>
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
      </figure>
    </section>

{dag_section}
  </main>

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
