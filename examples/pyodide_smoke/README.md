# Pyodide `pbt.run()` example

This is a minimal PyScript page that:

- loads the local `pbt` package into the browser
- sends an inline model to `await pbt.run(models_from_dict=...)`
- uses `MemoryStorageBackend` so SQLite is not required
- returns the run results in the page

The default model source is:

```jinja
{{ skip_and_set_to_value("") }}
```

That means the model executes through `pbt` without requiring any LLM call.

The example also loads the core Python dependencies used by `pbt`:

- `jinja2`
- `networkx`
- `rich`
- `click`
- `python-dotenv`

It does not try to run the full CLI or server stack. It is a browser-safe `await pbt.run()` example around `models_from_dict` using `pbt.storage.MemoryStorageBackend`.

## Run it

From the repo root:

```bash
python -m http.server 8000
```

Then open:

`http://localhost:8000/examples/pyodide_smoke/`

## What success looks like

The page should show:

- a green success message
- a textarea containing the inline `modelinclude` source
- a working `Run in pbt` button
- JSON output from `await pbt.run(models_from_dict={"modelinclude": ...})`

## What this is for

Use this as the first interactive browser check for `pbt`. If this page fails, the failure is in local package loading, PyScript configuration, or the browser-safe in-memory storage path.
