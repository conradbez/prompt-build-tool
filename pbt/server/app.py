"""
FastAPI application factory for the pbt server.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

try:
    from fastapi import FastAPI, Query
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "utils.server requires FastAPI and uvicorn. "
        "Install them with: pip install fastapi uvicorn"
    ) from exc

import pbt


class RunRequest(BaseModel):
    promptdata: dict[str, Any] | None = None
    select: list[str] | None = None


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


def _build_run_endpoint(models_dir: str, validation_dir: str, dag_promptdata: list[str]):
    """
    Dynamically build a /run function whose signature lists each detected
    promptdata() key as an optional query parameter. FastAPI reads __signature__
    to generate the OpenAPI schema, so every key shows up in /docs.
    """

    # The actual handler — receives promptdata values as keyword arguments
    def _run(**kwargs: Any) -> RunResponse:
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
        return RunResponse(outputs=serialised, errors=errors)

    # Build a signature: one Optional[str] query param per detected promptdata key
    params = [
        inspect.Parameter(
            key_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=Query(None, description=f"Template variable: `{{{{ promptdata('{key_name}') }}}}`"),
            annotation=Optional[str],
        )
        for key_name in dag_promptdata
    ]
    _run.__signature__ = inspect.Signature(params)

    description = (
        "Run all pbt prompt models and return their outputs.\n\n"
        + (
            "**Detected template variables** (from `promptdata()` usage in `.prompt` files):\n"
            + "\n".join(f"- `{v}`" for v in dag_promptdata)
            if dag_promptdata
            else "_No promptdata() variables detected in current models._"
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
    # Detect promptdata() keys at startup so the OpenAPI schema is accurate
    try:
        from pbt.graph import load_models, get_dag_promptdata
        models = load_models(models_dir)
        dag_promptdata = get_dag_promptdata(models)
    except Exception:
        dag_promptdata = []

    app = FastAPI(
        title="pbt server",
        description=(
            "Run pbt prompt models via HTTP. "
            "Query parameters on `/run` are auto-generated from `promptdata()` "
            "usage detected in your `.prompt` files."
        ),
        version=pbt.__version__,
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "pbt_version": pbt.__version__, "dag_promptdata": dag_promptdata}

    # POST /run — generic JSON body (for programmatic use)
    @app.post("/run", response_model=RunResponse, summary="Run models (JSON body)")
    def run_post(request: RunRequest) -> RunResponse:
        """Run pbt models with a JSON body. Useful for programmatic access."""
        try:
            outputs = pbt.run(
                models_dir=models_dir,
                select=request.select,
                promptdata=request.promptdata,
                validation_dir=validation_dir,
                verbose=False,
            )
        except Exception as exc:
            return RunResponse(outputs={}, errors=[str(exc)])
        serialised, errors = _serialise(outputs)
        return RunResponse(outputs=serialised, errors=errors)

    # GET /run — dynamic query params per detected promptdata key (for the docs UI)
    run_get = _build_run_endpoint(models_dir, validation_dir, dag_promptdata)
    app.get("/run", response_model=RunResponse, summary="Run models (query params)")(run_get)

    return app
