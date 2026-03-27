"""
FastAPI application factory for the pbt server.
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, File, Form, UploadFile
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "utils.server requires FastAPI and uvicorn. "
        "Install them with: pip install fastapi uvicorn"
    ) from exc

import pbt


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
    """
    Create and return a FastAPI app that exposes pbt over HTTP.

    The /run endpoint's query parameters are built dynamically from the vars
    detected across all .prompt files via static promptdata() scanning at startup.
    """
    app = FastAPI(
        title="pbt server",
        description="Run pbt prompt models via HTTP.",
        version=pbt.__version__,
    )

    # POST /run — multipart form: promptdata (JSON), select (JSON), optional file
    @app.post("/run", response_model=RunResponse, summary="Run models (form + file upload)")
    async def run_upload(
        promptdata: str | None = Form(None, description="JSON object of template variables, e.g. `{\"key\": \"value\"}`"),
        select: str | None = Form(None, description="JSON array of model names to run, e.g. `[\"model_a\"]`"),
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

        promptfiles = {"file": file.file} if file else None

        try:
            outputs = await pbt.run(
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

    return app
