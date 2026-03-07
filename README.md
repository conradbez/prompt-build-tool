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
  --models-dir TEXT    Directory containing *.prompt files  [default: models]
  --select / -s MODEL  Run only these models (and their dependencies).
                       Repeatable: -s outline -s article
  --no-color           Disable rich color output
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
    models_dir="models",   # path to *.prompt files
    select=["article"],    # optional: run only these models
    llm_call=my_llm_fn,    # optional: custom LLM backend
    rag_call=my_rag_fn,    # optional: custom RAG function
)
```

| Parameter | Type | Description |
|---|---|---|
| `models_dir` | `str` | Directory containing `*.prompt` files |
| `select` | `list[str] \| None` | Run only these models (upstream outputs loaded from DB) |
| `llm_call` | `(prompt: str) -> str \| None` | Override LLM backend. Falls back to `models/client.py` then Gemini |
| `rag_call` | `(*args) -> list \| str \| None` | Override RAG function. Falls back to `models/rag.py::do_RAG` |

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

## Project layout

```
prompt-build-tool-for-LLMs/
├── pbt/
│   ├── __init__.py      # package metadata
│   ├── cli.py           # Click CLI (pbt run, pbt ls, …)
│   ├── graph.py         # DAG builder + topological sort (networkx)
│   ├── parser.py        # Jinja2 renderer with ref() and return_list_RAG_results()
│   ├── executor.py      # LLM calls + SQLite writes
│   ├── llm.py           # LLM backend resolver (built-in Gemini or models/client.py)
│   ├── rag.py           # RAG resolver (models/rag.py → do_RAG)
│   └── db.py            # SQLite schema + query helpers
├── models/
│   ├── topic.prompt     # example: no dependencies
│   ├── outline.prompt   # example: depends on topic
│   ├── article.prompt   # example: depends on topic + outline
│   ├── client.py        # optional: custom LLM backend
│   └── rag.py           # optional: RAG function (do_RAG)
├── pyproject.toml
└── README.md
```

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required** (unless using `models/client.py`). Gemini API key. |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Override the Gemini model. |

---

## How dependency resolution works

1. pbt scans every `*.prompt` file for `ref('...')` calls using a regex.
2. It builds a directed acyclic graph (DAG) with [NetworkX](https://networkx.org/).
3. A topological sort gives the safe execution order.
4. If a model errors, all models that depend on it are marked **skipped** rather
   than failing with a confusing LLM error.
5. If a cycle is detected, pbt exits immediately with a clear error message.
