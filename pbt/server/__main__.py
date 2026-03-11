"""
Run the pbt server from the command line:

    python -m pbt.server --models-dir models --port 8000
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="pbt HTTP server")
    parser.add_argument("--models-dir", default="models", help="Directory with *.prompt files")
    parser.add_argument("--validation-dir", default="validation", help="Directory with validation *.py files")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required. Install with: pip install uvicorn")
        raise SystemExit(1)

    from pbt.server.app import create_app

    app = create_app(
        models_dir=args.models_dir,
        validation_dir=args.validation_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
