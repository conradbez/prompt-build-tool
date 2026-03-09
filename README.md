# pbt — prompt-build-tool

A **dbt-inspired** prompt orchestration tool for LLMs.

Write modular prompts in Jinja2, reference the output of other prompts with
`ref()`, and let **pbt** resolve dependencies. 

---


## Quick start

### 1. Install

```bash
pip install prompt-build-tool
```

### 2. Generate example

```bash
pbt init
# pbt init --provider openai
# pbt init --provider anthropic
# pbt init --provider gemini  (default)
```

### 3. Set your Gemini API key

```bash
export GEMINI_API_KEY=your_key_here
# export OPENAI_API_KEY=your_key_here
# export ANTHROPIC_API_KEY=your_key_here
```

Get a free key at <https://aistudio.google.com/app/apikey>.

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
import pbt

results = pbt.run("path/to/models")

for r in results:
    print(r.model_name, r.status, r.llm_output)
```

### `pbt.run()`

```python
pbt.run(
    models_dir="models",       # path to *.prompt files
    select=["article"],        # optional: run only these models
    llm_call=my_llm_fn,        # optional: custom LLM backend
    rag_call=my_rag_fn,        # optional: custom RAG function
    promptdata={"tone": "formal"},   # optional: variables injected via promptdata()
    validation_dir="validation", # optional: per-model validation functions
)
```

| Parameter | Type | Description |
|---|---|---|
| `models_dir` | `str` | Directory containing `*.prompt` files |
| `select` | `list[str] \| None` | Run only these models (upstream outputs loaded from DB) |
| `llm_call` | `(prompt: str) -> str \| None` | Override LLM backend. Falls back to `models/client.py` then Gemini |
| `rag_call` | `(*args) -> list \| str \| None` | Override RAG function. Falls back to `models/rag.py::do_RAG` |
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

## Customising the LLM backend (`models/client.py`)

Built to be unopinionated on how you do your LLM calls, by default pbt uses Gemini but expects you to implement your 
own LLM calls (usually 5 lines of code). To do so, edit or create `models/client.py` and define an `llm_call` function:

```python
# models/client.py
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

pbt will automatically detect and use this file instead of the built-in
Gemini implementation. If the file exists but does not define `llm_call`,
pbt raises an error at startup.

---

## RAG inside prompts (`models/rag.py`)

`pbt` has very little to say about RAG and leaves that up to you - you do this through the 
`return_list_RAG_results(*args)` function `pbt` give you access to in the .prompt template. `pbt` will pass this call to 
the  `do_RAG` function you define in  `models/rag.py`:

```python
# models/rag.py
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

If `models/rag.py` is absent and a template calls `return_list_RAG_results`,
pbt raises a clear error at render time.

---


## Passing files to models (`promptfiles`)

Models can receive files (PDFs, images, etc.) alongside the text prompt. Declare the files a model needs via `config()`, then provide the actual paths at runtime.

**1. Declare in config:**

```jinja
{{ config(promptfiles="my_document") }}
Summarise the attached document in 3 bullet points.
```

Multiple files are comma-separated:

```jinja
{{ config(promptfiles="report,chart_image") }}
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

Accept optional `files` and/or `config` parameters in your `models/client.py` — pbt passes them if the signature declares them:

```python
# models/client.py
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

## Validation (`validation/`)

Create a `validation/` directory with Python files matching model names. Each file must define `validate(prompt, result) -> bool`. If it returns `False`, the model is marked as an error and stops it use in downstream models.

```python
# validation/article.py
import json
from dataclasses import dataclass, fields


@dataclass
class Article:
    content: str
    author: str
    audience: str


def validate(prompt: str, result: str) -> bool:
    """Article output must be valid JSON matching the Article dataclass."""
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return False
    return all(
        f.name in data and isinstance(data[f.name], f.type)
        for f in fields(Article)
    ) and len(data.get("content", "")) >= 200
```

Run with `pbt run` — validation fires automatically after each model's LLM call.

### Typed hints in validation

Use Python dataclasses to define the expected shape of your model's JSON output. This makes validation self-documenting and easy to extend:

```python
# validation/summaries.py
import json
from dataclasses import dataclass, field, fields


@dataclass
class SummaryItem:
    title: str
    summary: str
    key_points: list[str]


@dataclass
class Summaries:
    summaries: list[SummaryItem] = field(default_factory=list)


def validate(prompt: str, result: str) -> bool:
    """Summaries output must be valid JSON matching the Summaries dataclass."""
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return False
    if not isinstance(data.get("summaries"), list) or not data["summaries"]:
        return False
    for item in data["summaries"]:
        if not all(f.name in item for f in fields(SummaryItem)):
            return False
        if not isinstance(item["key_points"], list) or len(item["key_points"]) < 1:
            return False
    return True
```

Dataclass-based validation pairs naturally with `{{ config(output_format="json") }}` models — the LLM is asked to return JSON, and the validator confirms the structure matches your expected schema before passing the result downstream.

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

## How to dynamically skip a model

If a rendered prompt evaluates to exactly `SKIP THIS MODEL`, pbt skips the LLM call and marks the model as `skipped`. Use the built-in `{{ skip_this_model }}` variable with a Jinja condition:

```jinja
{% if "no action needed" in ref('previous_model') %}
{{ skip_this_model }}
{% else %}
Summarise the following: {{ ref('previous_model') }}
{% endif %}
```

Downstream models that depend on a skipped model are also skipped automatically.
