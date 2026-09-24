"""
FastAPI application factory for the pbt server.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import HTMLResponse, Response
    from pydantic import BaseModel
    from jinja2 import Environment, FileSystemLoader
except ImportError as exc:
    raise ImportError(
        "utils.server requires FastAPI, uvicorn, and mako. "
        "Install them with: pip install fastapi uvicorn mako"
    ) from exc

import pbt
from pbt.files import contains_files, safe_name, to_jsonable, FileOutputError
from urllib.parse import quote


_TEMPLATES = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent.parent / "html_templates")),
    autoescape=True,
)


class _NamedBytesIO(io.BytesIO):
    """BytesIO with a .name attribute so LLM clients can call mimetypes.guess_type(f.name)."""

    def __init__(self, data: bytes, name: str) -> None:
        super().__init__(data)
        self.name = name


class RunResponse(BaseModel):
    outputs: dict[str, Any]
    errors: list[str] = []


def _with_urls(value: Any) -> Any:
    """Add a download ``url`` to every file manifest in a to_jsonable() value."""
    if isinstance(value, dict):
        out = {k: _with_urls(v) for k, v in value.items()}
        if out.get("type") == "file" and "sha256" in out:
            out["url"] = f"/blobs/{out['sha256']}?name={quote(str(out.get('name', 'file')))}"
        if out.get("type") == "dir" and isinstance(out.get("entries"), dict):
            for rel, entry in out["entries"].items():
                entry["url"] = f"/blobs/{entry['sha256']}?name={quote(rel.rsplit('/', 1)[-1])}"
        return out
    if isinstance(value, list):
        return [_with_urls(v) for v in value]
    return value


def _serialise(outputs: dict) -> tuple[dict[str, Any], list[str]]:
    serialised: dict[str, Any] = {}
    errors: list[str] = []
    for name, value in outputs.items():
        if isinstance(value, pbt.ModelStatus):
            serialised[name] = value.value
            errors.append(f"{name}: {value.value}")
        elif contains_files(value):
            # Files come back as manifests with a /blobs URL to fetch the bytes.
            serialised[name] = _with_urls(to_jsonable(value))
        else:
            serialised[name] = value
    return serialised, errors


def create_app(
    models_dir: str = "models",
    validation_dir: str = "validation",
) -> FastAPI:
    """Create and return a FastAPI app that exposes pbt over HTTP."""

    # Detect DAG shape at startup for the test UI
    try:
        from pbt.executor.graph import load_models, get_dag_promptdata, get_dag_promptfiles
        from pbt.llm import try_load_client_module

        # client.py registers any project-local model types; it has to be
        # imported before the models are parsed or those types look unknown.
        try_load_client_module(models_dir)
        _models = load_models(models_dir)
        dag_promptdata: list[str] = get_dag_promptdata(_models)
        dag_promptfiles: list[str] = get_dag_promptfiles(_models)
        model_names: list[str] = list(_models.keys())
    except Exception:
        dag_promptdata = []
        dag_promptfiles = []
        model_names = []

    app = FastAPI(
        title="pbt server",
        description="Run pbt prompt models via HTTP.",
        version=pbt.__version__,
    )

    def _blob_store():
        from pbt.llm import resolve_blob_store
        from pbt.storage.sqlite import SQLiteStorageBackend

        return resolve_blob_store(models_dir) or SQLiteStorageBackend().blob_store()

    # ------------------------------------------------------------------
    # GET /blobs/{sha256} — the bytes of a file a model produced
    # ------------------------------------------------------------------
    @app.get("/blobs/{sha256}", summary="Download a file a model produced")
    def get_blob(sha256: str, name: str = "file") -> Response:
        """Return a stored file's bytes, always as a download.

        Served as ``application/octet-stream`` with ``Content-Disposition:
        attachment`` whatever the file is, so a model-generated HTML or SVG
        file can never run as a page on this server's origin.
        """
        import re

        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise HTTPException(status_code=400, detail="Invalid sha256.")
        try:
            filename = safe_name(name)
        except FileOutputError:
            filename = "file"
        try:
            data = _blob_store().get(sha256)
        except KeyError:
            raise HTTPException(status_code=404, detail="No such blob.") from None
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
                "X-Content-Type-Options": "nosniff",
            },
        )

    # ------------------------------------------------------------------
    # POST /run — multipart form: promptdata (JSON), select (JSON), file
    # ------------------------------------------------------------------
    @app.post("/run", response_model=RunResponse, summary="Run models (form + file upload)")
    async def run_endpoint(
        promptdata: str | None = Form(None, description='JSON object of template variables, e.g. `{"key": "value"}`'),
        select: str | None = Form(None, description='JSON array of model names to run, e.g. `["model_a"]`'),
        file: UploadFile | None = File(None, description="Optional file passed as promptfile named 'file'"),
    ) -> RunResponse:
        """Run pbt models with optional form-encoded promptdata, select, and a file upload."""
        import json

        parsed_promptdata: dict | None = None
        if promptdata:
            try:
                parsed_promptdata = json.loads(promptdata)
            except json.JSONDecodeError as exc:
                return RunResponse(outputs={}, errors=[f"promptdata is not valid JSON: {exc}"])

        parsed_select: list[str] | None = None
        if select:
            try:
                parsed_select = json.loads(select)
            except json.JSONDecodeError as exc:
                return RunResponse(outputs={}, errors=[f"select is not valid JSON: {exc}"])

        promptfiles = {"file": _NamedBytesIO(await file.read(), file.filename)} if file else None

        try:
            outputs = await pbt.async_run(
                models_dir=models_dir,
                select=parsed_select,
                promptdata=parsed_promptdata,
                promptfiles=promptfiles,
                validation_dir=validation_dir,
                verbose=False,
            )
        except Exception as exc:
            return RunResponse(outputs={}, errors=[str(exc)])
        serialised, errors = _serialise(outputs)
        return RunResponse(outputs=serialised, errors=errors)

    # ------------------------------------------------------------------
    # GET /run — HTMX test UI
    # ------------------------------------------------------------------
    @app.get("/run", response_class=HTMLResponse, include_in_schema=False)
    def test_ui() -> str:
        return _TEMPLATES.get_template("run.html").render(
            dag_promptdata=dag_promptdata,
            dag_promptfiles=dag_promptfiles,
            model_names=model_names,
        )

    # ------------------------------------------------------------------
    # POST /run/execute — HTMX form handler, returns HTML fragment
    # ------------------------------------------------------------------
    @app.post("/run/execute", response_class=HTMLResponse, include_in_schema=False)
    async def test_run(request: Request) -> str:
        import html as _html

        form = await request.form()

        promptdata = {
            key: str(form[f"pd_{key}"])
            for key in dag_promptdata
            if form.get(f"pd_{key}")
        }

        promptfiles = {}
        for name in dag_promptfiles:
            upload = form.get(f"pf_{name}")
            if upload and getattr(upload, "filename", None):
                promptfiles[name] = _NamedBytesIO(await upload.read(), upload.filename)

        raw_select = form.getlist("select")
        selected = [s for s in raw_select if s] or None

        try:
            outputs = await pbt.async_run(
                models_dir=models_dir,
                select=selected,
                promptdata=promptdata or None,
                promptfiles=promptfiles or None,
                validation_dir=validation_dir,
                verbose=False,
            )
        except Exception as exc:
            return f'<article><p><mark>&#9888; {_html.escape(str(exc))}</mark></p></article>'

        serialised, errors = _serialise(outputs)

        parts = ['<h6>Model outputs</h6>']
        for model_name, output in serialised.items():
            is_error = isinstance(output, str) and output.startswith("error")
            if contains_files(outputs.get(model_name)):
                # The text a downstream model would see, file handles included.
                output = str(outputs[model_name])
            content = f'<ins>{_html.escape(str(output))}</ins>' if is_error else _html.escape(str(output))
            parts.append(
                f'<article>'
                f'<header><strong>{_html.escape(model_name)}</strong></header>'
                f'<pre><code>{content}</code></pre>'
                f'</article>'
            )

        for err in errors:
            parts.append(f'<article><p><mark>&#9888; {_html.escape(err)}</mark></p></article>')

        return "\n".join(parts) if parts else '<p><em>No outputs.</em></p>'

    return app
