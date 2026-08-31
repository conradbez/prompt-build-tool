# Defining a new model

Each file in this `models/` directory is one node in the execution DAG.
The file name (without the `.prompt` / `.prompt.jinja` suffix) becomes the
model name used by `ref()`.

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
| `output_extension` | `"md"` (default), e.g. `"html"` | File extension used when the CLI writes the output to disk |
| `model_type` | `"loop"`, `"execute_python"`, `"quality_check"` | Switch execution strategy (see below); omit for a plain LLM call |
| `promptfiles` | `["name"]` | Attach binary files (images, PDFs) passed via `--promptfile name=path` |
| `loop_over` | `"model_name"` | `loop` only — pin which dependency to fan out over |
| `quality_retries` | `"2"` (default) | `quality_check` only — how many check/retry rounds to generate |
| `quality_pass_marker` | `"PASS"` (default) | `quality_check` only — substring in the check output that means "good enough" |

Keys pbt does not recognise raise an `UnknownConfigKeyWarning` (with a
did-you-mean hint). If a custom `llm_call` consumes its own keys, declare them
once at import time with `pbt.register_config_keys("my_key")`.

## Referencing upstream models

Use `{{ ref('model_name') }}` anywhere in your prompt to inject a prior
model's output. These calls are also what build the DAG edges, so a model
depends on exactly the models it refs:

```
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
- With more than one such upstream, add `loop_over="topics"` to say which list
  to iterate.
- The output of the loop model is itself a JSON list, one entry per item.

## Execute-Python model

An `execute_python` model runs `exec()` on the rendered template instead of
calling the LLM. Use it for deterministic data transformation steps.

```
{{ config(model_type="execute_python") }}

data = ref('scores')           # list of numbers from an upstream model
output = sum(data) / len(data) # assign to `output` or use print()
```

The template is rendered first, then the result is exec'd. Upstream outputs are
available inside the code via `ref('model_name')` and the `model_outputs` dict.
Output is taken from stdout if anything was printed; otherwise from a variable
named `output` (`dict`/`list` are JSON-serialised, anything else via `str()`);
otherwise it is the empty string. **stdout wins over `output`.**

The code runs in-process with full builtins — there is no sandbox, so treat a
`.prompt` file as trusted code.

## Quality-check model

A `quality_check` model is expanded at DAG-build time into a chain of
check + retry nodes. Given `article → article_quality → summary` with
`quality_retries="2"`, the graph becomes:

```
article_quality_1   ← quality check on the original article
article_1           ← retry (skipped if check 1 passed)
article_quality_2   ← quality check on article_1
article_2           ← retry (skipped if check 2 passed)
article_quality     ← terminal pass-through, keeps the original name
```

The node's own template is the check prompt, and must `ref()` exactly one
upstream model — the one being checked. If its output contains
`quality_pass_marker` (default `"PASS"`), the following retry is skipped.

## Adding a new execution strategy

Model types are handler classes. Subclass `BaseModelHandler` (or an existing
handler, to inherit its caching/skip/JSON handling), set `model_type`, and
override `execute_node`:

```python
from pbt.executor.model_constructs import BaseModelHandler

class MyTypeHandler(BaseModelHandler):
    model_type = "my_type"

    async def execute_node(self, model_outputs, model_files, storage_backend,
                           run_id, llm_call, rag_call, promptdata,
                           prompt_skipped_models, skip_downstream_models,
                           validators=None):
        ...  # set model_outputs[self.name], return a ModelRunResult
```

To rewrite the graph instead of (or as well as) the execution — the way
`quality_check` does — override `inject_extra_nodes`.

There is currently **no public registration API**: the built-in types are wired
up in `_MODEL_CLASS_MAP` in `pbt/executor/graph.py`, and an unknown
`model_type` warns and falls back to a plain LLM call. Add your class to that
map (either by editing it, or from your own module before `execute_run()` is
called):

```python
from pbt.executor import graph
graph._MODEL_CLASS_MAP["my_type"] = MyTypeHandler
```
