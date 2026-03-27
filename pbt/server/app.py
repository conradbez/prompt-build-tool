"""
FastAPI application factory for the pbt server.
"""

from __future__ import annotations

import io
from typing import Any

try:
    from fastapi import FastAPI, File, Form, Request, UploadFile
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "utils.server requires FastAPI and uvicorn. "
        "Install them with: pip install fastapi uvicorn"
    ) from exc

import pbt

_PICO = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">'
_NAV = (
    '<nav class="container-fluid" style="position:sticky;top:0;z-index:999;'
    'border-bottom:1px solid var(--pico-muted-border-color)">'
    '<ul><li><strong>pbt</strong></li></ul>'
    '<ul>'
    '<li><a href="/test">Test runner</a></li>'
    '<li><a href="/docs-report">Docs</a></li>'
    '<li><a href="/docs" target="_blank">API docs ↗</a></li>'
    '</ul></nav>'
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
    # GET /test — HTMX test UI
    # ------------------------------------------------------------------
    @app.get("/test", response_class=HTMLResponse, include_in_schema=False)
    def test_ui() -> str:
        pd_fields = "".join(
            f"""
            <label for="pd_{key}">{key}
              <input type="text" id="pd_{key}" name="pd_{key}" placeholder="(optional)">
            </label>"""
            for key in dag_promptdata
        )

        pf_fields = "".join(
            f"""
            <label for="pf_{name}">{name} <small>file</small>
              <input type="file" id="pf_{name}" name="pf_{name}">
            </label>"""
            for name in dag_promptfiles
        )

        select_field = f"""
            <label for="select-models">Models
              <small>leave blank to run all</small>
              <select id="select-models" name="select" >
                {"".join(f'<option value="{n}">{n}</option>' for n in model_names)}
              </select>
            </label>""" if model_names else ""

        no_vars_msg = (
            "<p><em>No promptdata() or promptfile() variables detected in models.</em></p>"
            if not dag_promptdata and not dag_promptfiles else ""
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>pbt — test runner</title>
  {_PICO}
  <script src="https://unpkg.com/htmx.org@2.0.3/dist/htmx.min.js"></script>
  <style>
    #loader {{
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,.55); backdrop-filter: blur(3px);
      align-items: center; justify-content: center; z-index: 9999;
    }}
    #loader.htmx-request {{ display: flex; }}
    .spinner {{
      width: 48px; height: 48px;
      border: 4px solid var(--pico-muted-border-color);
      border-top-color: var(--pico-primary);
      border-radius: 50%; animation: spin .75s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    #results {{ margin-top: var(--pico-spacing); }}
  </style>
</head>
<body>
  <div id="loader"><div class="spinner"></div></div>
  {_NAV}
  <main class="container">
    <hgroup>
      <h1>pbt test runner</h1>
      <p>Run models and inspect their outputs.</p>
    </hgroup>

    <form hx-post="/test/run"
          hx-target="#results"
          hx-swap="innerHTML"
          hx-encoding="multipart/form-data"
          hx-indicator="#loader">
      {no_vars_msg}
      {pd_fields}
      {pf_fields}
      {select_field}
      <button type="submit">&#9654; Run</button>
    </form>

    <div id="results"></div>
  </main>
</body>
</html>"""

    # ------------------------------------------------------------------
    # POST /test/run — HTMX form handler, returns HTML fragment
    # ------------------------------------------------------------------
    @app.post("/test/run", response_class=HTMLResponse, include_in_schema=False)
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

        parts = []
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
