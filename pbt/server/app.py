"""
FastAPI application factory for the pbt server.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
    from jinja2 import Environment, FileSystemLoader
except ImportError as exc:
    raise ImportError(
        "utils.server requires FastAPI, uvicorn, and mako. "
        "Install them with: pip install fastapi uvicorn mako"
    ) from exc

import pbt


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


def _serialise(outputs: dict) -> tuple[dict[str, Any], list[str]]:
    serialised: dict[str, Any] = {}
    errors: list[str] = []
    for name, value in outputs.items():
        if isinstance(value, pbt.ModelStatus):
            serialised[name] = value.value
            errors.append(f"{name}: {value.value}")
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
