# Pyodide wheel smoke test

This example verifies the package in a browser as an installed artifact rather than as mounted source files.

It does three things:

- installs the built `prompt-build-tool` wheel from `dist/`
- imports `pbt` inside PyScript
- runs a minimal `await pbt.run(models_from_dict=...)` graph automatically with `MemoryStorageBackend`

There is no JS-to-Python button bridge. The page runs once at load time and writes the result to the page.

## Run it

From the repo root, rebuild the wheel so the browser test uses the current package contents:

```bash
uv build
python3 -m http.server 8000
```

Then open:

`http://localhost:8000/examples/pyodide_wheel_smoke/`

## What success looks like

The page should show:

- a green success message
- JSON output with `"passed": true`
- a `summary` result equal to `LLM::Return exactly the word browser`

## Why this exists

Use this to simulate `pip install prompt-build-tool` in Pyodide. If it fails while the source-mounted example still works, the problem is in packaging or wheel contents rather than the runtime itself.
