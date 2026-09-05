# pbt — prompt-build-tool

A **data-enineering-inspired** prompt orchestration tool for LLMs.

Write modular prompts in Jinja2, reference the output of other prompts with
`ref()`, and let **pbt** resolve dependencies. 

---


## Quick start

### 1. Install

```bash
pip install prompt-build-tool

# Also install the SDK for your LLM provider:
# pip install google-genai      # Gemini
# pip install openai            # OpenAI
# pip install anthropic         # Anthropic
```

### 2. Generate example

```bash
pbt init --provider anthropic
# pbt init --provider openai
# pbt init --provider gemini
```

### 3. Set your API key

```bash
export ANTHROPIC_API_KEY=your_key_here
# export OPENAI_API_KEY=your_key_here
# export GEMINI_API_KEY=your_key_here
```

### 4. Run

```bash
pbt run
```

### 5. Extend prompt models

In the `models/` directory write `.prompt` files:

```
models/
  topic.prompt
  outline.prompt
  article.prompt
```

Use `ref('model_name')` to inject the output of another model:

```jinja
{# models/outline.prompt #}
Based on this topic, create a detailed outline:

{{ ref('topic') }}
```

All standard Jinja2 syntax works too:

```jinja
{# models/comparison.prompt #}
{% set languages = ['Python', 'Go', 'Rust'] %}
Compare these languages for building CLI tools:
{% for lang in languages %}
- {{ lang }}
{% endfor %}

Context from previous analysis:
{{ ref('initial_analysis') }}
```

---


## Concepts (if you are familiar with data build tool)

| pbt concept | dbt analogy |
|---|---|
| `.prompt` file | `.sql` model file |
| `ref('model')` | `{{ ref('model') }}` |
| `models/` directory | `models/` directory |
| `global.prompt` | `query-comment` in `dbt_project.yml` |
| SQLite `runs` table | dbt `run_results.json` |
| SQLite `model_results` table | dbt `model` timing artifacts |

---


## Commands

### `pbt run`

Execute all prompt models in dependency order.

```
pbt run

```

### `pbt ls`

List discovered models and their dependency graph.

```bash
pbt ls
```


### `pbt test`

Run `tests/*.prompt` files against the latest run's outputs. Each test passes when the LLM returns `{"results": "pass"}`.

```bash
pbt test
```

**Inline params (`--promptdata` / `--promptfile`)** — pass params straight to `pbt test` to run the models with them and test against that run (an inline one-row `promptparams.csv`):

```bash
pbt test --promptdata tone=formal --promptfile doc=report.pdf
```

**Capture a run into `promptparams.csv` (`--add-to-csv`)** — when you find a param set worth keeping as a regression case, add `--add-to-csv` to append it as a new row in `promptparams.csv`, so it's re-tested in future parameterised runs:

```bash
pbt test --promptdata tone=formal --add-to-csv
```

`--add-to-csv` requires at least one `--promptdata`/`--promptfile`. Existing columns and rows are preserved; new columns are added to the header and older rows padded with empty cells.


### `pbt serve`

Start the pbt HTTP server and open the docs page in the browser.

```bash
pbt serve
# pbt serve --host 0.0.0.0 --port 8000
```


### `pbt docs`

Generate a self-contained HTML report of all previous runs with expandable model details and a DAG diagram.

```bash
pbt docs                        # writes to .pbt/docs/index.html
pbt docs --open                 # also opens in the browser
pbt docs --output my/report.html
```

---

## Python API

pbt can be used directly from Python without the CLI:

```python
import pbt

results = pbt.run("path/to/models")

for name, output in results.items():
    print(name, output)
```

### `pbt.run()`

```python
results = pbt.run(
    models_dir="models",       # path to *.prompt files
    select=["article"],        # optional: run only these models
    llm_call=my_llm_fn,        # optional: custom LLM backend
    rag_call=my_rag_fn,        # optional: custom RAG function
    promptdata={"tone": "formal"},   # optional: variables injected via promptdata()
    validation_dir="validation", # optional: per-model validation functions
    global_instruction="Answer in British English.",  # optional: text added to every prompt
)
```

| Parameter | Type | Description |
|---|---|---|
| `models_dir` | `str` | Directory containing `*.prompt` files |
| `select` | `list[str] \| None` | Run only these models (upstream outputs loaded from DB) |
| `llm_call` | `(prompt: str) -> str \| None` | Override LLM backend. Falls back to `client.py` (next to models/) |
| `rag_call` | `(*args) -> list \| str \| None` | Override RAG function. Falls back to `rag.py` (next to models/) `do_RAG` |
| `promptdata` | `dict \| None` | Variables injected into every template, accessed via `{{ promptdata('key') }}` |
| `promptfiles` | `dict \| None` | File paths by name, provided to models that declare `promptfiles:` in their config block |
| `validation_dir` | `str` | Directory with per-model `validate(prompt, result) -> bool` files |
| `global_instruction` | `str \| () -> str \| None` | Text rendered into every model's prompt. Falls back to `global.prompt` (next to models/) |

Returns a `dict` keyed by model name. Each value is the model's output string —
or `ModelStatus.SKIPPED` when an upstream model failed, or a `ModelError`
carrying the message when that model itself failed.

```python
results = pbt.run("models")

if isinstance(results["article"], pbt.ModelError):
    print("failed:", results["article"])
elif results["article"] is pbt.ModelStatus.SKIPPED:
    print("never ran")
else:
    print(results["article"])
```

`pbt.async_run(...)` takes the same arguments and returns the same dict, for
calling from inside an existing event loop.

---

## Passing variables to templates (`promptdata()`)

Inject runtime variables into templates using the `promptdata("name")` function — similar to how dbt's `source()` and `ref()` work.

```bash
pbt run --promptdata tone=formal --promptdata audience=engineers
```

```python
pbt.run("models", promptdata={"tone": "formal", "audience": "engineers"})
```

Access them in any `.prompt` file:

```jinja
Write an article in a {{ promptdata("tone") }} tone for {{ promptdata("audience") }}.

{% if promptdata("topic") %}
Topic: {{ promptdata("topic") }}
{% else %}
Choose a fascinating topic of your choice.
{% endif %}
```

`promptdata("name")` returns `None` if the variable was not provided, so `{% if promptdata("x") %}` is always safe.

---

## Customising the LLM backend (`client.py`)

pbt is unopinionated about which LLM you use. Create `client.py` at the project root (alongside your `models/` directory) and define an `llm_call` function — usually 5 lines:

```python
# client.py (Anthropic example)
import anthropic

def llm_call(prompt: str) -> str:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
```

pbt will automatically discover and use this file. Run `pbt init --provider <anthropic|openai|gemini>` to scaffold a starter `client.py` for your chosen provider. If the file exists but does not define `llm_call`, pbt raises an error at startup.

---

## RAG inside prompts (`rag.py`)

`pbt` has very little to say about RAG and leaves that up to you - you do this through the
`return_list_RAG_results(*args)` function `pbt` give you access to in the .prompt template. `pbt` will pass this call to
the `do_RAG` function you define in `rag.py` (at the project root, alongside your `models/` directory):

```python
# rag.py
def do_RAG(*args) -> list[str] | str:
    query = args[0]
    # your vector search, keyword lookup, etc.
    return ["Relevant document 1", "Relevant document 2"]
```

`do_RAG` receives whatever arguments you pass to `return_list_RAG_results`
in the template. It can return a `list[str]` or a bare `str` (wrapped
automatically). Return `False` or `None` to signal no results.

Use it in any `.prompt` file:

```jinja
{% set hits = return_list_RAG_results(ref('topic')) %}
{% if hits[0] %}
A related article in our library: "{{ hits[0] }}"

Write a paragraph explaining how the topic below connects to it:
{{ ref('topic') }}
{% else %}
Write a paragraph introducing this topic as a fresh subject:
{{ ref('topic') }}
{% endif %}
```

If `rag.py` is absent and a template calls `return_list_RAG_results`,
pbt raises a clear error at render time.

---


## Passing files to models (`promptfiles`)

Models can receive files (PDFs, images, etc.) alongside the text prompt. Declare the files a model needs via `config()`, then provide the actual paths at runtime.

**1. Declare in config:**

```jinja
{{ config(promptfiles=["my_document"]) }}
Summarise the attached document in 3 bullet points.
```

Multiple files use a JSON array:

```jinja
{{ config(promptfiles=["report", "chart_image"]) }}
```

**2. Provide file paths at runtime:**

```bash
pbt run --promptfile my_document=report.pdf
pbt run --promptfile report=annual.pdf --promptfile chart_image=q4.png
```

```python
pbt.run("models", promptfiles={"my_document": "report.pdf"})
pbt.run("models", promptfiles={"report": "annual.pdf", "chart_image": "q4.png"})
```

**3. Custom `llm_call` with file and config support:**

Accept optional `files` and/or `config` parameters in your `client.py` — pbt passes them if the signature declares them:

```python
# client.py
def llm_call(prompt: str, files: list[str] | None = None, config: dict | None = None) -> str:
    # files  — resolved file paths declared via config(promptfiles=...)
    # config — the full config dict for this model, e.g. {"output_format": "json"}
    ...
```

Both parameters are optional and independent — declare either, both, or neither.

---

## Output format config (`config()`)

Call `config()` at the top of a `.prompt` file to declare the expected output format:

```jinja
{{ config(output_format="json") }}
Return a JSON object with keys "title" and "summary".
```

When `output_format: json` is set, pbt validates the LLM output as JSON (stripping optional ` ```json ``` ` fences) and passes the parsed `dict`/`list` to downstream models via `ref()`, for example enabling `{{ ref('model').title }}` access.

### Recognised keys

| Key | Effect |
| --- | --- |
| `output_format` | `"json"` parses and validates the output as JSON; defaults to `"text"` |
| `output_extension` | File extension for `outputs/<model>.<ext>`; defaults to `"md"` |
| `promptfiles` | Names of files this model receives at runtime — see [Passing files to models](#passing-files-to-models-promptfiles) |
| `model_type` | `"template"`, `"loop"`, `"execute_python"`, `"quality_check"`, or a type you register; defaults to a plain LLM call |
| `loop_over` | For loop models: which upstream model to fan out over |
| `quality_retries` | For quality-check models: retry count (default `2`) |
| `quality_pass_marker` | Substring marking a passing quality check (default `"PASS"`) |
| `global_instruction` | `False` opts this model out of the run's [global instruction](#global-instructions-globalprompt) |

Any other key — and any unknown `model_type` — raises an `UnknownConfigKeyWarning` naming the model and file, with a did-you-mean suggestion, so typos like `output_fmt="json"` surface instead of being silently ignored. The key is still kept in the config dict, since pbt forwards the whole dict to a `llm_call(prompt, config=...)` that accepts one. If your `llm_call` consumes custom keys, register them once to silence the warning:

```python
import pbt

pbt.register_config_keys("temperature", "max_tokens")
```

---

## Global instructions (`global.prompt`)

Sometimes every prompt in a project needs the same preamble — a house style, a
persona, an output convention. This is pbt's analogue of dbt's `query-comment`:
one snippet, rendered into every prompt pbt sends.

Create a `global.prompt` next to your `models/` directory (the same place
`client.py` and `rag.py` live) and it is picked up automatically:

```
my_project/
├── client.py
├── global.prompt      ← rendered into every model's prompt
└── models/
    ├── article.prompt
    └── summary.prompt
```

```jinja
{# global.prompt #}
Write in British English. Never use em dashes.
```

By default the instruction is **prepended** to each model's prompt. Reference
`{{ prompt }}` to place the model body yourself:

```jinja
{# global.prompt — wrapper form #}
You are a careful technical writer.

<task>
{{ prompt }}
</task>

Answer in British English.
```

### It is a Jinja template too

The instruction is rendered with each model's own context, so
`{{ promptdata('key') }}`, `{{ model.name }}` and `{{ was_skipped('x') }}` all
work inside it:

```jinja
{% if promptdata("tone") %}Write in a {{ promptdata("tone") }} tone.{% endif %}
```

`ref()` is deliberately **not** available. The instruction goes into every
model, so a `ref('article')` would make every model depend on `article` —
including `article` itself. Use `promptdata()` for values that vary per run.

### Setting it from Python

```python
import pbt

pbt.run(global_instruction="Write in British English.")

# or build it at runtime — the callable is invoked once per run
pbt.run(global_instruction=lambda: load_house_style_from_somewhere())
```

The explicit argument wins over `global.prompt`. This is also the only way to
set one for `models_from_dict` runs, which never touch the filesystem. On the
CLI, `--global-instruction PATH` overrides the file for a single run:

```bash
pbt run --global-instruction experiments/terse.prompt
```

### Opting out

A single model opts out with `config()`:

```jinja
{{ config(global_instruction=False) }}
Return the raw JSON only, with no preamble.
```

Two exclusions are automatic:

- **`execute_python` models** never receive it — their template renders to
  Python source, and prepending prose to it would be a `SyntaxError`.
- **Test prompts** in `tests/` never receive it. The models under test render
  exactly as they do in a real run, but a judge told how to write is a biased
  judge.

Changing the instruction changes every rendered prompt, so the prompt cache
invalidates itself — the next `pbt run` re-runs affected models without
`--clear-cache`.

---

## Looping over a list (`model_type="loop"`)

Set `model_type="loop"` in `config()` to call the LLM once per item in an upstream list, then combine the results back into a list.

**1. Upstream model returns a JSON list:**

```jinja
{# models/articles.prompt #}
{{ config(output_format="json") }}
Return a JSON array of 3 article titles about {{ promptdata("topic") }}.
```

**2. Loop model processes each item:**

```jinja
{# models/summaries.prompt #}
{{ config(model_type="loop") }}

Write a one-paragraph summary for this article title:
{{ ref('articles') }}
```

`ref('articles')` returns the **current item** on each iteration — no new syntax needed.

**Result:** `summaries` outputs a JSON list with one entry per item from `articles`. Downstream models receive the full combined list via `ref('summaries')`.

**Multiple list dependencies** — if more than one upstream model returns a list, specify which to loop over:

```jinja
{{ config(model_type="loop", loop_over="articles") }}
```

---

## Template models (`model_type="template"`)

A `template` model renders its Jinja and uses the result as its output, with no
LLM call — for nodes that only reshape what upstream models already produced.

```jinja
{# models/report.prompt #}
{{ config(model_type="template") }}

# {{ ref('title') }}

{{ ref('summary') }}
```

---

## Python models (`model_type="execute_python"`)

An `execute_python` model runs its template as Python instead of sending it to
the LLM. Use it for the deterministic steps in a pipeline — counting, parsing,
reshaping, arithmetic on an upstream result.

```jinja
{# models/length.prompt #}
{{ config(model_type="execute_python") }}
output = len(ref('article').split())
```

The template is rendered first, then the result is executed. Inside the code,
`ref('name')` returns an upstream model's output and `model_outputs` holds them
all. Those `ref()` calls are also what build the dependency edges, exactly as in
a normal prompt.

The output is whatever the code prints; if it prints nothing, a variable named
`output` is used instead (`dict` and `list` are JSON-encoded, anything else via
`str()`). Printing wins over `output`.

Results are cached on the rendered code, so unchanged code does not re-run. The
code executes in-process with full builtins and no sandbox, so treat a `.prompt`
file as trusted code.

---

## Validation (`validation/`)

Create a `validation/` directory with Python files matching model names. Each file must define `validate(prompt, result) -> bool`. If it returns `False`, the model is marked as an error and stops it use in downstream models.

```python
# validation/article.py
import json
from pydantic import BaseModel, ValidationError


class Article(BaseModel):
    content: str
    author: str
    audience: str


def validate(prompt: str, result: str) -> bool:
    """Article output must be valid JSON matching the Article model."""
    try:
        data = json.loads(result)
        article = Article(**data)
    except (json.JSONDecodeError, ValidationError):
        return False
    return len(article.content) >= 200
```

Run with `pbt run` — validation fires automatically after each model's LLM call.

---

## HTTP server (`utils/server`)

Deploy over to run and return LLM response to .prompt pipeline over HTTP. Runs a lightweight FastAPI server and manages pipeline execution and return (requires `pip install fastapi uvicorn`):

```bash
python -m utils.server --models-dir models --port 8000
```

```
POST /run   body: {"promptdata": {"tone": "formal"}, "select": ["article"]}
            returns: {"outputs": {"topic": "...", "article": "..."}}

GET  /health
```

Or use the factory in Python:

```python
from utils.server import create_app
import uvicorn

app = create_app(models_dir="models")
uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Quality checks with automatic retries (`model_type="quality_check"`)

Add a quality-check node after any model to automatically retry it until it passes (or a retry limit is reached).

**1. Write the quality check:**

```jinja
{# models/article_quality.prompt #}
{{ config(model_type="quality_check", quality_retries="2") }}
Does this article have a clear introduction, body, and conclusion?
Reply PASS if yes, FAIL and explain why if not.

Article: {{ ref('article') }}
```

**2. The upstream model can use the feedback:**

```jinja
{# models/article.prompt #}
{% if model.meta.feedback_from_previous_run %}
A previous attempt was rejected. Feedback: {{ model.meta.feedback_from_previous_run }}
Rewrite the article addressing that feedback.
{% else %}
Write an article about {{ ref('topic') }}.
{% endif %}
```

pbt expands `article_quality` into an interleaved retry chain at run time:

```
article             ← original run
article_quality_1   ← quality check (contains PASS or FAIL + reason)
article_1           ← retry (skipped if quality_1 passed)
article_quality_2   ← quality check on article_1
article_2           ← retry (skipped if quality_2 passed)
article_quality     ← terminal pass-through (output of best attempt)
```

Downstream models depend on `article_quality` as normal — they always receive the best passing output.

**Config options:**

| Option | Default | Description |
|---|---|---|
| `quality_retries` | `"2"` | Number of retry attempts |
| `quality_pass_marker` | `"PASS"` | Substring to detect in quality check output to mark success |

---

## How to dynamically skip a model

Use `{{ skip_and_set_to_value("value") }}` to skip the LLM call during Jinja rendering and provide the output directly:

```jinja
{% if "no action needed" in ref('previous_model') %}
{{ skip_and_set_to_value("No action needed.") }}
{% else %}
Summarise the following: {{ ref('previous_model') }}
{% endif %}
```

The model is recorded as a successful run, downstream templates can detect it with `was_skipped('model_name')`, and downstream `ref()` calls receive the value you provided.

---

## Writing your own model kind (`model_type=`)

Every `.prompt` file is run by a *model kind*. Leave `model_type` unset and pbt
sends the rendered prompt to your LLM; set it to `loop`, `template`,
`execute_python` or `quality_check` and pbt runs it differently. If none of
those do what you need, you can add your own.

A kind is a function and a registration, both in `client.py`:

```python
# client.py
import pbt

@pbt.model_kind("shout", config_keys={"suffix"})
async def shout(rendered, call):
    response = await call.llm(rendered)
    return response.upper() + call.spec.config.get("suffix", "")
```

Then use it from any model:

```jinja
{# models/loud.prompt #}
{{ config(model_type="shout", suffix="!") }}
Summarise {{ ref('article') }}.
```

```bash
pbt run
```

### What you get

pbt renders the template for you and hands your function two things:

| | |
|---|---|
| `rendered` | The rendered prompt text, with every `ref()`, `promptdata()` and skip function already resolved |
| `call` | `call.llm(rendered)` sends it to your LLM; `call.spec` is the model being run (`call.spec.name`, `call.spec.config`); `call.outputs` holds upstream outputs by name; `call.compute(rendered, compute=fn)` caches arbitrary work |

Whatever you return becomes the model's output — the thing `ref('loud')` gives
downstream models, and the thing written to `outputs/`. Return a string and it
is parsed for you if the model sets `output_format="json"`. Return a list or a
dict and it is kept as-is.

Everything else keeps working without you doing anything: the prompt cache,
`{{ config(output_format="json") }}`, the skip functions, `validation/`,
`pbt test`, `pbt docs` and the run report all treat your kind like a built-in
one. `call.llm` is already cached, timed and skip-aware — there is nothing to
opt into.

`config_keys={"suffix"}` tells pbt which `config()` keys your kind reads.
Without it, `pbt run` warns that `suffix` looks like a typo.

### The full record

The decorator is shorthand for building a `ModelKind` and registering it. The
long form is what you want for a kind that has no `exec_fn` of its own:

```python
pbt.register_model_kind(pbt.ModelKind(
    name="shout",
    exec_fn=shout,
    config_keys={"suffix"},
))
```

| Field | Default | Meaning |
|---|---|---|
| `name` | — | The `config(model_type=...)` value |
| `exec_fn` | `None` | `async (rendered, call) -> Any`. `None` means the rendered text *is* the output |
| `fan_out` | `False` | Render once per item of an upstream JSON list, run `exec_fn` on each concurrently |
| `expand_fn` | `None` | `(spec, all_specs) -> list[ModelSpec] \| None` — rewrite this node into several |
| `config_keys` | `frozenset()` | The `config()` keys this kind reads |
| `accepts_global_instruction` | `True` | `False` when the rendered text is not a prompt for a model to answer |

The five built-ins are nothing but this record. `template` is
`ModelKind("template", exec_fn=None, accepts_global_instruction=False)`; `loop`
is the plain LLM call with `fan_out=True`; `quality_check` is an `expand_fn` and
nothing else.

### Optional extras

**Skip the LLM entirely.** Anything you can compute, you can return:

```python
@pbt.model_kind("truncate", config_keys={"max_words"})
async def truncate(rendered, call):
    return " ".join(rendered.split()[:call.spec.config_int("max_words", 50)])
```

```jinja
{{ config(model_type="truncate", max_words="30") }}
{{ ref('article') }}
```

The `ref()` calls in the template are what tell pbt your model has to run after
the models it references — reading `call.outputs` directly does not create that
edge.

For a *one-off* calculation, you do not need a kind at all:
[`execute_python`](#python-models-model_typeexecute_python) already runs a
model's template as Python. Write a kind when the behaviour is worth reusing
across models and configuring per model, the way `truncate` takes `max_words`.

**Cache expensive non-LLM work.** Anything slow and repeatable can go behind the
same cache your LLM calls use, so it does not re-run when nothing changed:

```python
return await call.compute(rendered, compute=lambda: scrape(rendered))
```

**Fan out over a list.** Set `fan_out=True` and pbt finds the upstream
dependency whose output is a JSON list, renders your template once per item
(with `ref()` on that model yielding the current item), runs your `exec_fn` on
each concurrently, and collects the results into a list. That is all `loop` is.

**Opt out of the global instruction.** If your rendered template is not a prompt
for a model to answer — Python source, or a value passed straight through — set
`accepts_global_instruction=False` so your
[global instruction](#global-instructions-globalprompt) is not prepended to it.

**Turn one model into several.** Give the kind an `expand_fn(spec, all_specs)`
to replace your node with a chain of nodes before the run starts — this is how
`quality_check` builds its check-and-retry loop. Return a list of models, one of
which must keep the original name so `ref()` still finds it.

### Where to register it

Anywhere that runs before your models are read. `client.py` is the easiest spot,
because pbt already imports it on every `pbt run`, `pbt test` and `pbt ls`.

A worked example lives in [`examples/custom_model_type/`](examples/custom_model_type/).

Upgrading from 0.3, where a custom type was a class? See
[docs/MIGRATION_model_kinds.md](docs/MIGRATION_model_kinds.md).
