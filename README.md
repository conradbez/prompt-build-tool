# pbt — prompt-build-tool

A **dbt-inspired** prompt orchestration tool for LLMs.

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
import asyncio
import pbt

results = asyncio.run(pbt.run("path/to/models"))

for name, output in results.items():
    print(name, output)
```

### `pbt.run()`

```python
import asyncio

results = asyncio.run(pbt.run(
    models_dir="models",       # path to *.prompt files
    select=["article"],        # optional: run only these models
    llm_call=my_llm_fn,        # optional: custom LLM backend
    rag_call=my_rag_fn,        # optional: custom RAG function
    promptdata={"tone": "formal"},   # optional: variables injected via promptdata()
    validation_dir="validation", # optional: per-model validation functions
))
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

Returns a list of `ModelRunResult` objects with fields: `model_name`, `status`, `prompt_rendered`, `llm_output`, `error`, `execution_ms`, `cached`.

---

## Passing variables to templates (`promptdata()`)

Inject runtime variables into templates using the `promptdata("name")` function — similar to how dbt's `source()` and `ref()` work.

```bash
pbt run --promptdata tone=formal --promptdata audience=engineers
```

```python
asyncio.run(pbt.run("models", promptdata={"tone": "formal", "audience": "engineers"}))
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
asyncio.run(pbt.run("models", promptfiles={"my_document": "report.pdf"}))
asyncio.run(pbt.run("models", promptfiles={"report": "annual.pdf", "chart_image": "q4.png"}))
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
