"""
FastAPI application factory for the pbt server.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, List, Optional

try:
    from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "utils.server requires FastAPI and uvicorn. "
        "Install them with: pip install fastapi uvicorn"
    ) from exc

import pbt

# Maps output_extension config values to HTTP Content-Type headers.
_EXTENSION_CONTENT_TYPE: dict[str, str] = {
    "html": "text/html; charset=utf-8",
    "json": "application/json",
    "md":   "text/markdown; charset=utf-8",
    "txt":  "text/plain; charset=utf-8",
    "csv":  "text/csv; charset=utf-8",
    "xml":  "application/xml; charset=utf-8",
}


def _raw_response(
    serialised: dict[str, Any],
    output_model: str | None,
    model_extensions: dict[str, str],
) -> "Response | None":
    """Return a raw Response with the correct Content-Type when ``output_model``
    is set and that model has ``output_extension`` configured.  Returns None
    otherwise (caller should fall back to the normal RunResponse JSON path)."""
    if output_model is None or output_model not in model_extensions:
        return None
    if output_model not in serialised:
        return None
    ext = model_extensions[output_model]
    content_type = _EXTENSION_CONTENT_TYPE.get(ext, "text/plain; charset=utf-8")
    return Response(content=str(serialised[output_model]), media_type=content_type)


class RunResponse(BaseModel):
    outputs: dict[str, Any]
    errors: list[str] = []


class _DagNode(BaseModel):
    name: str
    source: str


class _DagSubmit(BaseModel):
    nodes: List[_DagNode]


class _DagResponse(BaseModel):
    dag_id: str


class _DagRunRequest(BaseModel):
    select: Optional[List[str]] = None
    promptdata: Optional[dict] = None


def _filter_output(serialised: dict[str, Any], output_model: str | None) -> dict[str, Any]:
    """Return only the requested model's output, or all outputs if not specified."""
    if output_model is None:
        return serialised
    if output_model not in serialised:
        raise KeyError(output_model)
    return {output_model: serialised[output_model]}


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


def _parse_promptdata(promptdata_json: str | None) -> dict | None:
    """Parse the JSON-encoded promptdata form field, or return None."""
    if not promptdata_json:
        return None
    try:
        parsed = json.loads(promptdata_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"promptdata must be a valid JSON object, got: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("promptdata must be a JSON object (dict), not a list or scalar.")
    return parsed or None


def _parse_promptfiles(uploads: list[UploadFile] | None) -> dict | None:
    """
    Convert a list of uploaded files into the ``promptfiles`` dict expected by
    ``pbt.run()``.  Each file's ``filename`` (without extension) is used as the
    promptfile key, so the caller should name files to match the keys declared
    in the model's ``# pbt:config promptfiles:`` block.

    Example: upload a file named ``doc.pdf`` → key ``"doc"``.
    If the filename has no extension the full filename is used as the key.
    """
    if not uploads:
        return None
    result: dict = {}
    for upload in uploads:
        raw_name = upload.filename or ""
        # Strip extension to get the key (e.g. "doc.pdf" → "doc")
        key = raw_name.rsplit(".", 1)[0] if "." in raw_name else raw_name
        if not key:
            key = raw_name
        result[key] = upload.file  # SpooledTemporaryFile satisfies IO[bytes]
    return result or None


def _build_run_endpoint(
    models_dir: str,
    validation_dir: str,
    dag_promptdata: list[str],
    dag_promptfiles: list[str],
    model_extensions: dict[str, str] | None = None,
):
    """
    Dynamically build a /run function whose signature lists each detected
    promptdata() key as an optional query parameter. FastAPI reads __signature__
    to generate the OpenAPI schema, so every key shows up in /docs.

    promptfiles cannot be passed via GET (no file uploads), but they are
    documented in the endpoint description.
    """

    _model_extensions = model_extensions or {}

    def _run(**kwargs: Any) -> RunResponse:
        output_model = kwargs.pop("output_model", None)
        provided = {k: v for k, v in kwargs.items() if v is not None}
        try:
            outputs = pbt.run(
                models_dir=models_dir,
                promptdata=provided or None,
                validation_dir=validation_dir,
                verbose=False,
            )
        except Exception as exc:
            return RunResponse(outputs={}, errors=[str(exc)])
        serialised, errors = _serialise(outputs)
        raw = _raw_response(serialised, output_model, _model_extensions)
        if raw is not None:
            return raw
        try:
            serialised = _filter_output(serialised, output_model)
        except KeyError:
            return RunResponse(outputs={}, errors=[f"output_model '{output_model}' not found in run results"])
        return RunResponse(outputs=serialised, errors=errors)

    # Build a signature: one Optional[str] query param per detected promptdata key,
    # plus a fixed output_model param.
    params = [
        inspect.Parameter(
            key_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=Query(None, description=f"Template variable: `{{{{ promptdata('{key_name}') }}}}`"),
            annotation=Optional[str],
        )
        for key_name in dag_promptdata
    ]
    params.append(
        inspect.Parameter(
            "output_model",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=Query(None, description="If provided, only return this model's output."),
            annotation=Optional[str],
        )
    )
    _run.__signature__ = inspect.Signature(params)

    promptfiles_note = (
        "\n\n**Required promptfiles** (upload via `POST /run`):\n"
        + "\n".join(f"- `{v}`" for v in dag_promptfiles)
        if dag_promptfiles
        else ""
    )

    description = (
        "Run all pbt prompt models and return their outputs.\n\n"
        + (
            "**Detected template variables** (from `promptdata()` usage in `.prompt` files):\n"
            + "\n".join(f"- `{v}`" for v in dag_promptdata)
            if dag_promptdata
            else "_No promptdata() variables detected in current models._"
        )
        + promptfiles_note
        + (
            "\n\n> **Note:** File uploads (`promptfiles`) are only supported via `POST /run`."
            if dag_promptfiles
            else ""
        )
    )
    _run.__doc__ = description

    return _run


def create_app(
    models_dir: str = "models",
    validation_dir: str = "validation",
) -> FastAPI:
    """
    Create and return a FastAPI app that exposes pbt over HTTP.

    The /run endpoint's query parameters are built dynamically from the vars
    detected across all .prompt files via static promptdata() scanning at startup.
    """
    # Import DAG helpers once at create_app() time — avoids repeated per-request
    # import overhead and surfaces import errors at startup rather than on first call.
    from pbt.executor.graph import (
        load_models, get_dag_promptdata, get_dag_promptfiles,
        build_models_from_dict, compute_dag_hash, models_to_json,
    )
    from pbt import db as _db

    # Initialise SQLite schema once at startup (idempotent).
    _db.init_db()

    # Detect promptdata() keys and promptfile names at startup so the OpenAPI
    # schema is accurate.
    try:
        models = load_models(models_dir)
        dag_promptdata = get_dag_promptdata(models)
        dag_promptfiles = get_dag_promptfiles(models)
        model_extensions = {
            name: m.config["output_extension"]
            for name, m in models.items()
            if "output_extension" in m.config
        }
    except Exception:
        dag_promptdata = []
        dag_promptfiles = []
        model_extensions = {}

    app = FastAPI(
        title="pbt server",
        description=(
            "Run pbt prompt models via HTTP. "
            "Query parameters on `/run` are auto-generated from `promptdata()` "
            "usage detected in your `.prompt` files. "
            "File inputs (`promptfiles`) are accepted as multipart uploads on `POST /run`."
        ),
        version=pbt.__version__,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "pbt_version": pbt.__version__,
            "dag_promptdata": dag_promptdata,
            "dag_promptfiles": dag_promptfiles,
        }

    # POST /run — multipart/form-data so that both text variables (promptdata)
    # and file uploads (promptfiles) can be included in the same request.
    #
    # promptdata  : JSON-encoded dict, e.g. '{"country": "USA"}'
    # select      : repeated form field, e.g. select=tweet&select=haiku
    # <any file>  : uploaded file; filename (without extension) is the promptfile key
    @app.post("/run", response_model=RunResponse, summary="Run models (multipart form)")
    async def run_post(
        promptdata: Optional[str] = Form(
            None,
            description=(
                "JSON-encoded dict of template variables, e.g. "
                '`{"country": "USA", "tone": "formal"}`. '
                "Matches `promptdata()` calls in `.prompt` templates."
            ),
        ),
        select: Optional[List[str]] = Form(
            None,
            description="Limit execution to these model names (and their upstream dependencies).",
        ),
        output_model: Optional[str] = Form(
            None,
            description="If provided, only return this model's output instead of all outputs.",
        ),
        promptfiles: Optional[List[UploadFile]] = File(
            None,
            description=(
                "Files required by models that declare `promptfiles` in their config block. "
                "Each file's **filename** (without extension) is used as the promptfile key, "
                "so name your upload to match the key declared in the model "
                "(e.g. upload `doc.pdf` for `promptfiles: doc`)."
                + (
                    f" Detected keys in current models: {', '.join(f'`{v}`' for v in dag_promptfiles)}."
                    if dag_promptfiles
                    else ""
                )
            ),
        ),
    ) -> RunResponse:
        """
        Run pbt models. Accepts multipart/form-data so file inputs (promptfiles)
        can be uploaded alongside text variables (promptdata).

        **promptdata** — JSON object string with template variables.\n
        **select** — repeated field to limit which models run.\n
        **output_model** — if set, only this model's output is returned.\n
        **promptfiles** — one file per declared promptfile key; use the key as the filename
        (with any extension), e.g. `doc.pdf` → key `doc`.
        """
        try:
            pd = _parse_promptdata(promptdata)
        except ValueError as exc:
            return RunResponse(outputs={}, errors=[str(exc)])

        pf = _parse_promptfiles(promptfiles)

        try:
            outputs = pbt.run(
                models_dir=models_dir,
                select=select,
                promptdata=pd,
                promptfiles=pf,
                validation_dir=validation_dir,
                verbose=False,
            )
        except Exception as exc:
            return RunResponse(outputs={}, errors=[str(exc)])
        serialised, errors = _serialise(outputs)
        raw = _raw_response(serialised, output_model, model_extensions)
        if raw is not None:
            return raw
        try:
            serialised = _filter_output(serialised, output_model)
        except KeyError:
            return RunResponse(outputs={}, errors=[f"output_model '{output_model}' not found in run results"])
        return RunResponse(outputs=serialised, errors=errors)

    # GET /run — dynamic query params per detected promptdata key (for the docs UI).
    # File uploads are not possible via GET; use POST /run for promptfiles.
    run_get = _build_run_endpoint(models_dir, validation_dir, dag_promptdata, dag_promptfiles, model_extensions)
    app.get("/run", response_model=RunResponse, summary="Run models (query params)")(run_get)

    # -----------------------------------------------------------------------
    # DAG endpoints — used by the drag-and-drop front-end (utils/dnd-front-end)
    # -----------------------------------------------------------------------

    @app.post(
        "/dag",
        response_model=_DagResponse,
        summary="Register a DAG from the front-end editor",
        tags=["DAG editor"],
    )
    def submit_dag(body: _DagSubmit) -> _DagResponse:
        """
        Receive a list of nodes ``{name, source}`` from the drag-and-drop UI,
        build a PromptModel DAG, persist it in the database, and return the
        stable ``dag_id`` that the front-end can use for subsequent ``/run``
        calls.

        The ``source`` field of each node is a raw Jinja2 prompt template
        (identical to a ``.prompt`` file) including any ``ref()`` and
        ``promptdata()`` calls.
        """
        models_dict = {node.name: node.source for node in body.nodes}
        try:
            models_map = build_models_from_dict(models_dict)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        dag_hash = compute_dag_hash(models_map)
        _db.save_dag(dag_hash, models_to_json(models_map))
        return _DagResponse(dag_id=dag_hash)

    @app.post(
        "/dag/{dag_id}/run",
        response_model=RunResponse,
        summary="Run one or more models from a registered DAG",
        tags=["DAG editor"],
    )
    async def run_dag(
        dag_id: str,
        select: Optional[List[str]] = Form(
            None,
            description="Model names to run (and their upstream dependencies). Repeat for multiple.",
        ),
        promptdata: Optional[str] = Form(
            None,
            description='JSON-encoded dict of template variables, e.g. `{"topic": "AI"}`.',
        ),
        promptfiles: Optional[List[UploadFile]] = File(
            None,
            description=(
                "Files required by models that declare `promptfiles` in their config block. "
                "Each file's filename (without extension) is used as the promptfile key."
            ),
        ),
    ) -> RunResponse:
        """
        Execute the models listed in ``select`` (and their upstream dependencies)
        from the DAG previously registered via ``POST /dag``.

        Accepts **multipart/form-data** so that both runtime variables (``promptdata``)
        and file inputs (``promptfiles``) can be sent in the same request.
        """
        try:
            pd = _parse_promptdata(promptdata)
        except ValueError as exc:
            return RunResponse(outputs={}, errors=[str(exc)])

        pf = _parse_promptfiles(promptfiles)

        try:
            outputs = pbt.run(
                dag_id=dag_id,
                select=select or None,
                promptdata=pd,
                promptfiles=pf,
                verbose=False,
            )
        except Exception as exc:
            return RunResponse(outputs={}, errors=[str(exc)])
        serialised, errors = _serialise(outputs)
        return RunResponse(outputs=serialised, errors=errors)

    return app
