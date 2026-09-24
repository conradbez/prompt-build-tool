"""
pbt docs — generate a self-contained HTML report of all previous runs.

The generated HTML includes:
  - A summary table of every pbt run (status, model count, timing)
  - Expandable per-run model results
  - A Mermaid.js DAG diagram of the current model dependency graph
  - The files models produced: a gallery for the latest run, and previews and
    downloads beside each model's output

Files are copied out of the blob store into ``files/`` next to the HTML file
(``files/<sha256>/<name>``) and linked relatively, so the report stays a
folder you can open, zip or serve as it is.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader

from pbt.files import (
    BlobStore,
    FileOutputError,
    contains_files,
    decode_output,
    display_text,
    iter_files,
)

if TYPE_CHECKING:
    from pbt.files import File
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


def _mermaid_dag(models: dict[str, "ModelSpec"], file_models: "set[str] | None" = None) -> str:
    """Return a Mermaid flowchart string for the model DAG.

    Models in *file_models* produced files in the latest run; they are drawn
    as documents, and an edge that attaches files (``promptfiles``) rather
    than referencing text is dotted.
    """
    file_models = file_models or set()
    lines = ["graph LR"]
    for name in sorted(models):
        safe = name.replace("-", "_")
        if name in file_models:
            lines.append(f"    {safe}[/\"{name} 📎\"/]:::files")
        else:
            lines.append(f"    {safe}[{name}]")
    for name, model in sorted(models.items()):
        safe_dst = name.replace("-", "_")
        attached = {f.split(".")[0] for f in model.promptfiles_used}
        for dep in model.depends_on:
            safe_src = dep.replace("-", "_")
            arrow = "-. files .->" if dep in attached else "-->"
            lines.append(f"    {safe_src} {arrow} {safe_dst}")
    if file_models:
        lines.append("    classDef files fill:#e0f2fe,stroke:#0284c7")
    return "\n".join(lines)


#: Image types previewed inline.  Everything else is a download link.
PREVIEW_MIMES = frozenset({
    "image/png", "image/jpeg", "image/gif", "image/webp", "image/avif",
    "image/bmp", "image/svg+xml",
})


class _FileExporter:
    """Copies files out of the blob store into the report's ``files/`` folder.

    Each blob is written at most once, however many runs reference it.
    """

    def __init__(self, blob_store: BlobStore | None, root: Path) -> None:
        self.blob_store = blob_store
        self.root = root
        self._done: dict[tuple[str, str], str | None] = {}

    def decode(self, raw: str | None):
        try:
            return decode_output(raw, self.blob_store)
        except FileOutputError:
            return raw

    def describe(self, value) -> list[dict]:
        """One dict per file in *value*, for the template."""
        return [self._one(label, file) for label, file in iter_files(value)]

    def _one(self, label: str, file: "File") -> dict:
        key = (file.sha256, file.name)
        if key not in self._done:
            self._done[key] = self._write(file)
        href = self._done[key]
        return {
            "label": label,
            "name": file.name,
            "mime": file.mime,
            "size": _human_size(file.size),
            "sha": file.sha256[:12],
            "href": href,
            "missing": href is None,
            "preview": href is not None and file.mime in PREVIEW_MIMES,
        }

    def _write(self, file: "File") -> str | None:
        target = self.root / "files" / file.sha256 / file.name
        if not target.exists():
            try:
                file.save(target.parent)
            except FileOutputError:
                return None  # blob missing or corrupt: listed, not linked
        return f"files/{file.sha256}/{quote(file.name)}"


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024 or unit == "MB":
            return f"{size} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size} B"  # pragma: no cover


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
    blob_store: BlobStore | None = None,
) -> None:
    """Write the HTML docs file to *output_path*.

    Files the models produced are copied from *blob_store* into a ``files/``
    folder beside it.
    """
    exporter = _FileExporter(blob_store, output_path.parent)

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

    def _result(r) -> dict:
        value = exporter.decode(_model_output(r))
        files = exporter.describe(value) if contains_files(value) else []
        text = display_text(value) if files else (value if isinstance(value, str) else "")
        return {
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
            "output": (text or "")[:_FULL_LIMIT],
            "output_preview": (text or "")[:200],
            "files": files,
            "cached": bool(_column(r, "cached")),
        }

    runs_data = []
    for run in runs:
        rid = run["run_id"]
        results = [_result(r) for r in run_results.get(rid, [])]
        runs_data.append({
            "run_id": rid,
            "short_id": rid[:8] + "…",
            "status": run["status"] or "—",
            "model_count": run["model_count"] or 0,
            "created": (run["created_at"] or "")[:19].replace("T", " "),
            "duration": _duration(run["created_at"], run["completed_at"]),
            "results": results,
        })

    # The newest run's files, gallery-first: what the pipeline makes, at a glance.
    latest = runs_data[0] if runs_data else None
    latest_files = [
        {**f, "model_name": r["model_name"]}
        for r in (latest["results"] if latest else [])
        for f in r["files"]
    ]
    file_models = {r["model_name"] for r in (latest["results"] if latest else []) if r["files"]}

    html = _TEMPLATES.get_template("docs.html").render(
        runs=runs_data,
        latest_run=latest,
        latest_files=latest_files,
        status_colours=STATUS_COLOURS,
        dag_section=_mermaid_dag(models, file_models) if models else "",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
