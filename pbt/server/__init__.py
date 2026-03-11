"""
pbt server — run pbt models via a lightweight FastAPI HTTP server.

Usage
-----
    from pbt.server import create_app
    import uvicorn

    app = create_app(models_dir="models")
    uvicorn.run(app, host="0.0.0.0", port=8000)

Or from the command line::

    python -m pbt.server --models-dir models --port 8000

API
---
POST /run
    Body (JSON): { "vars": { "key": "value" }, "select": ["model_a"] }
    Returns (JSON): { "outputs": { "model_name": "..." }, "run_id": "..." }

GET /health
    Returns: { "status": "ok" }
"""

from pbt.server.app import create_app

__all__ = ["create_app"]
