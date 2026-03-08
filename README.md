# pbt — prompt-build-tool

A **dbt-inspired** prompt orchestration tool for LLMs.

Write modular prompts in Jinja2, reference the output of other prompts with
`ref()`, and let **pbt** resolve dependencies, execute everything in order via
Gemini, and store every input/output in a SQLite database for full auditability.

---

## Concepts

| pbt concept | dbt analogy |
|---|---|
| `.prompt` file | `.sql` model file |
| `ref('model')` | `{{ ref('model') }}` |
| `models/` directory | `models/` directory |
| SQLite `runs` table | dbt `run_results.json` |
| SQLite `model_results` table | dbt `model` timing artifacts |

---

## Quick start

### 1. Install

```bash
pip install -e .
```

### 2. Set your Gemini API key

```bash
export GEMINI_API_KEY=your_key_here
```

Get a free key at <https://aistudio.google.com/app/apikey>.

### 3. Add prompt models

Create a `models/` directory and write `.prompt` files:

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

### 4. Run

```bash
pbt run
```

---

## Commands

### `pbt run`

Execute all prompt models in dependency order.

```
pbt run [OPTIONS]

Options:
  --models-dir TEXT       Directory containing *.prompt files  [default: models]
  --select / -s MODEL     Run only these models (and their dependencies).
                          Repeatable: -s outline -s article
  --promptdata KEY=VALUE  Inject a variable via promptdata() into every template.
                          Repeatable: --promptdata country=USA --promptdata tone=formal
  --promptfile NAME=PATH  Provide a file for models that declare it in config.
                          Repeatable: --promptfile doc=report.pdf --promptfile img=chart.png
  --validation-dir TEXT   Directory with per-model validation Python files  [default: validation]
  --no-color              Disable rich color output
```

Example output:

```
─────────────────── pbt run ───────────────────
  Run ID  : 3f2a1b4c-...
  Models  : 3

  [1/3] topic   … OK (1 204 ms)
  [2/3] outline … OK (2 891 ms)
  [3/3] article … OK (5 102 ms)

────────────────────────────────────────────────
  Done  : 3 succeeded
  Run ID: 3f2a1b4c-...
  DB    : .pbt/pbt.db
```

### `pbt ls`

List discovered models and their dependency graph.

```bash
pbt ls
```

### `pbt docs`

Generate a self-contained HTML report of all previous runs with expandable model details and a DAG diagram.

```bash
pbt docs                        # writes to .pbt/docs/index.html
pbt docs --open                 # also opens in the browser
pbt docs --output my/report.html
```

### `pbt test`

Run `tests/*.prompt` files against the latest run's outputs. Each test passes when the LLM returns `{"results": "pass"}`.

```bash
pbt test
pbt test --run-id <run_id>
```

### `pbt show-runs`

Show recent run history from the SQLite store.

```bash
pbt show-runs --limit 20
```

### `pbt show-result MODEL_NAME`

Print the stored input/output for a model.

```bash
pbt show-result article              # latest run
pbt show-result article --show all   # rendered prompt + LLM output
pbt show-result article --run-id <run_id>
```

---

## SQLite schema

All results are stored in `.pbt/pbt.db`.

### `runs`

One row per `pbt run` invocation.

| Column | Type | Description |
|---|---|---|
| `run_id` | TEXT PK | UUID for the run |
| `created_at` | TIMESTAMP | When the run started |
| `status` | TEXT | `running` / `success` / `error` / `partial` |
| `completed_at` | TIMESTAMP | When the run finished |
| `model_count` | INTEGER | Number of models in the run |
| `git_sha` | TEXT | Short git SHA (if in a git repo) |

### `model_results`

One row per model per run.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `run_id` | TEXT FK | Parent run |
| `model_name` | TEXT | Stem of the `.prompt` file |
| `status` | TEXT | `pending` / `running` / `success` / `error` / `skipped` |
| `prompt_template` | TEXT | Raw `.prompt` file contents |
| `prompt_rendered` | TEXT | Fully-rendered prompt sent to the LLM |
| `llm_output` | TEXT | Raw LLM response text |
| `started_at` | TIMESTAMP | Execution start |
| `completed_at` | TIMESTAMP | Execution end |
| `execution_ms` | INTEGER | Wall-clock time in milliseconds |
| `error` | TEXT | Error message if status = `error` |
| `depends_on` | TEXT | JSON list of upstream model names |

Query results directly:

```bash
sqlite3 .pbt/pbt.db "SELECT model_name, status, execution_ms FROM model_results ORDER BY id DESC LIMIT 10"
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

### Passing functions inline

```python
import anthropic
import pbt

def my_llm(prompt: str) -> str:
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

def my_rag(*args) -> list[str]:
    query = args[0]
    # your vector search here
    return ["Relevant doc 1", "Relevant doc 2"]

results = pbt.run("models", llm_call=my_llm, rag_call=my_rag)
```

---

## Customising the LLM backend (`models/client.py`)

By default pbt uses Gemini. To swap in any other LLM, create
`models/client.py` and define an `llm_call` function:

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

pbt exposes a `return_list_RAG_results(*args)` Jinja function in every
template. To power it, create `models/rag.py` with a `do_RAG` function:

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

## Passing files to models (`promptfiles`)

Models can receive files (PDFs, images, etc.) alongside the text prompt. Declare the files a model needs in its `{# pbt:config #}` block, then provide the actual paths at runtime.

**1. Declare in config block:**

```jinja
{# pbt:config
promptfiles: my_document
#}
Summarise the attached document in 3 bullet points.
```

Multiple files are comma-separated:

```jinja
{# pbt:config
promptfiles: report, chart_image
#}
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

**3. Custom `llm_call` with file support:**

To handle files in your own `models/client.py`, accept an optional `files` parameter:

```python
# models/client.py
def llm_call(prompt: str, files: list[str] | None = None) -> str:
    # files is a list of resolved file paths for this model
    ...
```

pbt checks the function signature at runtime — if `files` is not in the signature, files are silently omitted (backward compatible).

---

## Output format config (`{# pbt:config #}`)

Add an optional config block at the top of a `.prompt` file to declare the expected output format:

```jinja
{# pbt:config
output_format: json
#}
Return a JSON object with keys "title" and "summary".
```

When `output_format: json` is set, pbt validates the LLM output as JSON (stripping optional ` ```json ``` ` fences) and passes the parsed `dict`/`list` to downstream models via `ref()`, enabling `{{ ref('model').title }}` access.

---

## Validation (`validation/`)

Create a `validation/` directory with Python files matching model names. Each file must define `validate(prompt, result) -> bool`. If it returns `False`, the model is marked as an error.

```python
# validation/article.py
def validate(prompt: str, result: str) -> bool:
    return len(result) > 100  # require at least 100 characters
```

Run with `pbt run` — validation fires automatically after each model's LLM call.

---

## HTTP server (`utils/server`)

Run pbt over HTTP with a lightweight FastAPI server (requires `pip install fastapi uvicorn`):

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

## Project layout

```
prompt-build-tool-for-LLMs/
├── pbt/
│   ├── __init__.py      # Python API (pbt.run)
│   ├── cli.py           # Click CLI (pbt run, pbt test, pbt docs, …)
│   ├── graph.py         # DAG builder + topological sort (networkx)
│   ├── parser.py        # Jinja2 renderer with ref(), config block parsing
│   ├── executor.py      # LLM calls + SQLite writes + validation hooks
│   ├── llm.py           # LLM backend resolver (built-in Gemini or models/client.py)
│   ├── rag.py           # RAG resolver (models/rag.py → do_RAG)
│   ├── db.py            # SQLite schema + query helpers
│   ├── docs.py          # HTML report generator (pbt docs)
│   ├── tester.py        # Test runner (pbt test)
│   └── validator.py     # Validation framework (validation/*.py)
├── models/
│   ├── topic.prompt     # example: no dependencies
│   ├── outline.prompt   # example: depends on topic
│   ├── article.prompt   # example: depends on topic + outline
│   ├── client.py        # optional: custom LLM backend
│   └── rag.py           # optional: RAG function (do_RAG)
├── validation/          # optional: per-model validate(prompt, result)->bool files
├── utils/
│   └── server/          # FastAPI HTTP server (POST /run, GET /health)
├── pyproject.toml
└── README.md
```

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required** (unless using `models/client.py`). Gemini API key. |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Override the Gemini model. |

---

## How dependency resolution works

1. pbt scans every `*.prompt` file for `ref('...')` calls using a regex.
2. It builds a directed acyclic graph (DAG) with [NetworkX](https://networkx.org/).
3. A topological sort gives the safe execution order.
4. If a model errors, all models that depend on it are marked **skipped** rather
   than failing with a confusing LLM error.
5. If a cycle is detected, pbt exits immediately with a clear error message.

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
