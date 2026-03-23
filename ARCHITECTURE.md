# pbt — Architecture & Developer Notes

---

## Module map

```
pbt/
  __init__.py          Python API (pbt.run) — also resolves llm/rag backends
  cli.py               Click commands — orchestrates discovery, calls execute_run
  executor/
    graph.py           PromptModel dataclass, DAG building, topological sort, serialisation
    parser.py          Jinja2 rendering, config() parsing
    executor.py        Pure execution loop — no file discovery, no CLI concerns
    model_constructs.py  Specialised model_type handlers (loop, …) imported by executor
  llm.py / rag.py / validator.py  Backend resolvers
  db.py                SQLite schema + queries
  tester.py / docs.py  pbt test and pbt docs implementations
```

---

## Key design decisions

**`executor.py` is a pure executor.** It takes callables (`llm_call`, `rag_call`, `validators`) and never touches the filesystem. File discovery lives in `cli.py` and `pbt/__init__.py`. This keeps the executor testable with mock callables.

**DAG hash covers structure and content.** `compute_dag_hash()` hashes model names, dependency edges, prompt source text, and config. Any change to structure or prompt content produces a new hash. The hash is also the stable key used to look up the DAG snapshot in the `dags` table.

**DAG snapshots are persisted.** After each run, the full DAG (all model sources, configs, and edges) is stored in the `dags` table keyed by `dag_hash`. Pass `dag_id=<hash>` to `pbt.run()` or `--dag-id <hash>` to `pbt run` to replay a specific DAG version from DB without reading *.prompt files from disk.

**Prompt cache is content-addressed.** SHA256 of the *rendered* prompt (post-Jinja, pre-LLM) is the cache key. Identical rendered prompts across any run reuse the stored output.

**`--select` runs the full upstream chain fresh.** `pbt run --select tweet` runs `tweet` and all its ancestors in dependency order. The prompt cache makes unchanged upstream nodes instant — no stale-output risk, no need for a previous run of the same DAG.

**`model_outputs` is `dict[str, str | dict | list]`.** When a model declares `output_format: json`, its entry is a parsed Python object, not a string. Downstream `ref('model')` in Jinja receives the object, enabling `{{ ref('model').key }}` access. The DB always stores canonical JSON strings.

**Validators must not return the raw `result` string when `output_format: json` is set.** The executor parses the LLM output into a Python `dict`/`list` before running validators, storing it in `model_outputs`. If the validator returns the string `result` unchanged, it overwrites the parsed object with a string — breaking any downstream loop model that expects a list. Passthrough validators should either `return True` or return the already-parsed value (e.g. `json.loads(result)`). Validators that don't need to transform JSON output should simply be omitted.

**`model_type` constructs live in `model_constructs.py`.** Specialised execution strategies (e.g. `loop`, `template`) are async functions in `pbt/executor/model_constructs.py`, each with the signature `execute_*(model, model_outputs, ...) -> ModelRunResult`. They are registered in `CONSTRUCT_REGISTRY` (a plain dict at the bottom of that module). The executor looks up the handler via `CONSTRUCT_REGISTRY.get(model_type)` — no changes to `executor.py` are needed when adding a new construct. Adding a new construct means adding one function and one entry in `CONSTRUCT_REGISTRY`, both in `model_constructs.py`.

---

## Static promptdata() detection

Before any run, pbt scans each template with a regex to discover which `promptdata()` keys it uses — no Jinja rendering needed.

```python
_PROMPTDATA_PATTERN = re.compile(r"""\bpromptdata\(\s*['"](\w+)['"]\s*\)""")

def detect_used_promptdata(template_source: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _PROMPTDATA_PATTERN.finditer(template_source):
        seen[match.group(1)] = None
    return list(seen)
```

Results stored in `PromptModel.promptdata_used`, shown in `pbt ls`, and warned about in `pbt run` if not provided. This is simpler and more reliable than the previous VarSpy dry-render approach — all branches of conditionals are detected since it's a static scan.

---

## Validation vs. tests

| | `tests/` | `validation/` |
|---|---|---|
| **Format** | `.prompt` Jinja files | `.py` Python files |
| **When** | Explicit `pbt test` after a run | Automatically inside `pbt run` |
| **Input** | Model outputs from a previous run | Rendered prompt + LLM output of the current model |
| **Pass criterion** | LLM returns `{"results": "pass"}` | `validate(prompt, result) -> bool` |
| **On failure** | Non-zero exit from `pbt test` | Model marked `error`, downstream skipped |

---

## SQLite

All results are stored in `.pbt/pbt.db`.

### SQLite notes

- DB at `.pbt/pbt.db` relative to cwd.
- `PRAGMA journal_mode=WAL` — allows concurrent readers during a run.
- `_migrate()` applies idempotent `ALTER TABLE ADD COLUMN` for backward compat.
- `prompt_hash` is indexed for cache lookups; `dag_hash` is indexed on `runs` for test-run matching.
- `dags` table stores one row per unique DAG content hash; `INSERT OR IGNORE` keeps it idempotent.



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
| `promptfiles` | `dict \| None` | File paths by name, provided to models that declare `promptfiles:` via `config()` |
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

results = asyncio.run(pbt.run("models", llm_call=my_llm, rag_call=my_rag))
```

---

## Project layout

```
prompt-build-tool-for-LLMs/
├── pbt/
│   ├── __init__.py      # Python API (pbt.run)
│   ├── cli.py           # Click CLI (pbt run, pbt test, pbt docs, …)
│   ├── executor/
│   │   ├── graph.py     # DAG builder + topological sort (networkx) + serialisation
│   │   ├── parser.py    # Jinja2 renderer with ref(), config() parsing
│   │   └── executor.py  # LLM calls + SQLite writes + validation hooks
│   ├── llm.py           # LLM backend resolver (loads client.py)
│   ├── rag.py           # RAG resolver (rag.py → do_RAG)
│   ├── db.py            # SQLite schema + query helpers
│   ├── docs.py          # HTML report generator (pbt docs)
│   ├── tester.py        # Test runner (pbt test)
│   └── validator.py     # Validation framework (validation/*.py)
├── client.py            # optional: custom LLM backend
├── rag.py               # optional: RAG function (do_RAG)
├── models/
│   ├── topic.prompt     # example: no dependencies
│   ├── outline.prompt   # example: depends on topic
│   └── article.prompt   # example: depends on topic + outline
├── validation/          # optional: per-model validate(prompt, result)->bool files
├── utils/
│   └── server/          # FastAPI HTTP server (POST /run, GET /health)
├── pyproject.toml
└── README.md
```

---

---

## How dependency resolution works

1. pbt scans every `*.prompt` file for `ref('...')` calls using a regex.
2. It builds a directed acyclic graph (DAG) with [NetworkX](https://networkx.org/).
3. A topological sort gives the safe execution order.
4. If a model errors, all models that depend on it are marked **skipped** rather
   than failing with a confusing LLM error.
5. If a cycle is detected, pbt exits immediately with a clear error message.

---
