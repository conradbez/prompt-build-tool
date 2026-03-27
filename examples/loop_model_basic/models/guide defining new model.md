# Defining a new model

Each file in this `models/` directory is one node in the execution DAG.
The file name (without extension) becomes the model name used by `ref()`.

## Minimal model

```
models/my_model.prompt
```

```
Write a one-sentence summary of the history of the internet.
```

No configuration needed — the model runs against the LLM and its output is
available downstream as `{{ ref('my_model') }}`.

## Config block

Place a `{{ config(...) }}` call at the top of the file to set options:

```
{{ config(output_format="json") }}

Return a JSON array of 5 trending topics in AI. Return only the array.
```

| Option | Values | Effect |
|---|---|---|
| `output_format` | `"text"` (default), `"json"` | Parse output as JSON; downstream `ref()` receives a Python object |
| `model_type` | `"loop"`, `"execute_python"` | Switch execution strategy (see below) |
| `promptfiles` | `["name"]` | Attach binary files (images, PDFs) passed via `--promptfile name=path` |

## Referencing upstream models

Use `{{ ref('model_name') }}` anywhere in your prompt to inject a prior
model's output:

```
{{ config(output_format="json") }}

Summarise this article in three bullet points:

{{ ref('article') }}
```

## Loop model

A `loop` model fans out over a JSON list produced by an upstream model,
calling the LLM once per item in parallel and collecting results into a list.

```
{{ config(model_type="loop") }}

Write a 2–3 sentence summary of this technology topic:

{{ ref('topics') }}
```

- The upstream model (`topics`) must have `output_format="json"` and return a
  JSON array.
- The output of the loop model is itself a JSON list, one entry per item.

## Execute-Python model

An `execute_python` model runs `exec()` on the rendered template instead of
calling the LLM. Use it for deterministic data transformation steps.

```
{{ config(model_type="execute_python") }}

data = ref('scores')           # list of numbers from an upstream model
output = sum(data) / len(data) # assign to `output` or use print()
```

Upstream outputs are available via `ref('model_name')` and the
`model_outputs` dict. Write to stdout (`print(...)`) or assign to a variable
named `output`; the last one wins.

## Adding a new execution strategy

To add a model type beyond `loop` and `execute_python`, register an
`execute_node` callback in `pbt/executor/model_type_registry.py`:

```python
from pbt.executor.model_type_registry import register_execute_node_callback

async def execute_my_type(model, model_outputs, model_files,
                          storage_backend, run_id, llm_call, rag_call,
                          promptdata, prompt_skipped_models,
                          parse_json_output):
    ...  # mutate model_outputs[model.name], return ModelRunResult

register_execute_node_callback("my_type", execute_my_type)
```

Import your module anywhere before `execute_run()` is called (e.g. at the
bottom of the file that defines the handler) and the executor will route
`model_type="my_type"` models to it automatically — no changes to the
executor needed.
