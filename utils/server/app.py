"""
FastAPI application factory for the pbt server.
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "utils.server requires FastAPI and uvicorn. "
        "Install them with: pip install fastapi uvicorn"
    ) from exc

import pbt


class RunRequest(BaseModel):
    vars: dict[str, Any] | None = None
    select: list[str] | None = None


class RunResponse(BaseModel):
    run_id: str | None = None
    outputs: dict[str, Any]
    errors: list[str] = []


def create_app(
    models_dir: str = "models",
    validation_dir: str = "validation",
) -> FastAPI:
    """
    Create and return a FastAPI app that exposes pbt over HTTP.

    Parameters
    ----------
    models_dir:
        Path to the directory containing *.prompt files.
    validation_dir:
        Path to the directory containing per-model validation Python files.
    """
    app = FastAPI(
        title="pbt server",
        description="Run pbt prompt models via HTTP",
        version=pbt.__version__,
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "pbt_version": pbt.__version__}

    @app.post("/run", response_model=RunResponse)
    def run(request: RunRequest) -> RunResponse:
        """
        Run pbt models and return their outputs.

        - **vars**: Variables injected into every Jinja2 template (like `--var` on CLI)
        - **select**: Run only these models (reuses upstream outputs from latest run)
        """
        errors: list[str] = []

        try:
            outputs = pbt.run(
                models_dir=models_dir,
                select=request.select,
                vars=request.vars,
                validation_dir=validation_dir,
                verbose=False,
            )
        except Exception as exc:
            return RunResponse(outputs={}, errors=[str(exc)])

        # Serialise outputs: convert ModelStatus enums to their string values,
        # keep dict/list (from json output_format models) as-is.
        serialised: dict[str, Any] = {}
        for name, value in outputs.items():
            if isinstance(value, pbt.ModelStatus):
                serialised[name] = value.value
                errors.append(f"{name}: {value.value}")
            else:
                serialised[name] = value

        return RunResponse(outputs=serialised, errors=errors)

    return app
