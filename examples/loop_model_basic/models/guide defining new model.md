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
| `model_type` | `"template"`, `"loop"`, `"execute_python"`, `"quality_check"`, or one you register | Switch execution strategy (see below); omit for a plain LLM call |
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

## Template model

A `template` model renders its Jinja and uses the result as the output — no LLM
call. Use it to reshape or combine what upstream models already produced.

```
{{ config(model_type="template") }}

# {{ ref('title') }}

{{ ref('body') }}
```

## Adding a new execution kind

A model kind is a frozen record of data with, at most, one function. Write the
function and register it:

```python
# client.py — pbt imports this before parsing your models
import pbt

@pbt.model_kind("shout", config_keys={"suffix"})
async def shout(rendered, call):
    response = await call.llm(rendered)
    return response.upper() + call.spec.config.get("suffix", "")
```

```
{{ config(model_type="shout", suffix="!") }}
Summarise {{ ref('article') }}.
```

pbt renders the template before calling you. `rendered` is that text; `call`
has the model already bound to it — `call.llm(rendered)` sends the prompt,
`call.spec` is the parsed model (`name`, `source`, `config`, `depends_on`), and
`call.outputs` holds what upstream models produced.

You only write that one step. The prompt cache, `output_format="json"` parsing,
skip functions, validation and storage are applied to your kind exactly as they
are to the built-in ones. `config_keys` declares the `config()` keys your kind
reads, so they stop being reported as typos.

The other fields on `ModelKind`, for what a plain function cannot express:

| Field | Purpose |
|---|---|
| `exec_fn=None` | The rendered text *is* the output — no LLM call. This is all `template` is |
| `fan_out=True` | Render once per item of an upstream JSON list and run concurrently. This is all `loop` is |
| `accepts_global_instruction=False` | Your rendered template is not a natural-language prompt, so the run's global instruction must not be prepended |
| `expand_fn=(spec, all_specs)` | Rewrite this node into several nodes at DAG-build time, the way `quality_check` does. Return a list of specs including exactly one keeping the declared name |

For work that is expensive but not an LLM call, run it through
`call.compute(rendered, compute=...)` so it gets the same prompt cache — that is
how `execute_python` avoids re-running unchanged code.
