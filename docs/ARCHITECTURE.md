# pbt — Architecture & Developer Notes

---

## Module map

```
pbt/
  __init__.py          Python API (pbt.run) — also resolves llm/rag backends
  cli.py               Click commands — orchestrates discovery, calls execute_run
  executor/
    graph.py           DAG building, spec parsing, model-type expansion
    parser_initial.py  Static analysis — extract_dependencies, parse_model_config
    parser_model.py    Jinja2 rendering (render_prompt), skip-function helpers
    executor.py        Pure execution loop — no file discovery, no CLI concerns
    run_context.py     RunContext — run-wide inputs, rendering, the prompt cache
    builtin_types.py   The built-in model types (llm, template, loop, execute_python, quality_check)
    model_constructs.py  Deprecated aliases for the pre-registry handler classes
  model_spec.py        ModelSpec — the parsed, inert description of one model
  model_types.py       ModelType protocol + the public registry
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

**Model data, model behaviour, and run inputs are three separate things.** A `ModelSpec` (`pbt/model_spec.py`) is inert parsed data: source, deps, config, `model_type`. A `ModelType` (`pbt/model_types.py`) is a stateless strategy with one required method, `async execute(spec, ctx)`. A `RunContext` (`pbt/executor/run_context.py`) carries every run-wide input — the backends, promptdata, promptfiles, validators, the global instruction, accumulated outputs — plus the shared mechanisms `render()` and `call_llm()`. Keeping them apart is why a new run-wide input is one field on `RunContext` rather than a new parameter on six functions, and why a new model type is one class rather than an edit in the parser, the graph, and the executor.

**`config()` keys are validated, not enforced.** `known_config_keys()` in `parser_initial.py` unions three sources: `_BUILTIN_CONFIG_KEYS` (the keys pbt acts on for every model), the `config_keys` each registered model type declares, and anything passed to `pbt.register_config_keys(...)` for a custom `llm_call`. A key outside all three (or an unregistered `model_type`) triggers an `UnknownConfigKeyWarning` at load time naming the model and file. It stays a warning because the whole config dict is forwarded to any `llm_call` accepting a `config` parameter, so custom keys are legitimate. **A type-specific config key belongs on that type's `config_keys`**, not in `_BUILTIN_CONFIG_KEYS` — registering the type is then the only step needed.

**`ModelType.expand()` enables DAG-build-time node expansion.** After each model is parsed, `graph.py` calls its type's `expand(spec, all_specs)`. Returning a list of specs replaces that one node with all of them, before the DAG is built; returning `None` leaves it alone. The list must contain exactly one node keeping the declared name, so downstream `ref()` calls still resolve — `graph.py` raises if it does not. `all_specs` is read-only, for looking up already-parsed upstream models. `quality_check` uses this to expand one node into an interleaved chain of check and retry nodes, terminating in a `template` node that carries the original name.

**The executor owns the lifecycle; a type owns one step.** `execute_model()` in `executor.py` resolves the strategy, asks it for a value, then applies skip propagation, `output_format` parsing, storage, and validation — identically for built-in and user-registered types. A strategy that wants caching calls `ctx.cached(rendered, spec, state, compute=...)`, which is the single caching path: `call_llm` is a thin wrapper over it, and `execute_python` uses it directly so unchanged code does not re-execute.

**Storage keeps raw and validated output separately.** `mark_model_success` stores the raw output, which is what the prompt cache serves, so editing a validator never forces a new LLM call. When a validator transforms the output, the executor also calls `record_validated_output`, and `get_model_outputs_from_run` prefers it — so `pbt test` judges the value the pipeline actually passed downstream.

---

## Adding a model type

One class and one registration. Everything else — caching, skipping, JSON
parsing, validation, storage — is the executor's job and already applies.

```python
import pbt

@pbt.model_type("shout", config_keys={"suffix"})
class Shout(pbt.BaseModelType):
    async def execute(self, spec, ctx):
        rendered, state = ctx.render(spec)
        output = await ctx.call_llm(rendered, spec, state)
        return output.upper() + spec.config.get("suffix", "")
```

```jinja
{# models/loud.prompt #}
{{ config(model_type="shout", suffix="!") }}
Summarise {{ ref('article') }}.
```

The class may also set `accepts_global_instruction = False` (for a type whose
rendered template is not a natural-language prompt) and override
`expand(spec, all_specs)` to rewrite itself into several nodes.

Put the registration in `client.py`: pbt imports it before parsing any
`.prompt` file, precisely so project-local types are available. Registering
anywhere that runs earlier works too.

Return value of `execute`: a `str` is parsed when the model sets
`output_format="json"`; anything else (a list, a dict) is stored as-is, which is
how a fan-out returns per-item results.

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
- `init_db()` is `CREATE TABLE IF NOT EXISTS` only — there is no migration path, so delete `.pbt/pbt.db` after a schema change.
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
| `llm_output` | TEXT | Raw LLM response text (what the prompt cache serves) |
| `llm_output_validated` | TEXT | Post-validation output, when a validator changed it |
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

results = pbt.run("models", llm_call=my_llm, rag_call=my_rag)
```

---

## Project layout

```
prompt-build-tool-for-LLMs/
├── pbt/
│   ├── __init__.py      # Python API (pbt.run)
│   ├── cli.py           # Click CLI (pbt run, pbt test, pbt docs, …)
│   ├── executor/
│   │   ├── graph.py          # DAG builder, spec parsing, model-type expansion
│   │   ├── parser_initial.py # Static analysis: deps, config, promptdata extraction
│   │   ├── parser_model.py   # Jinja2 render_prompt, ref(), skip helpers
│   │   ├── run_context.py    # RunContext: run-wide inputs, render, prompt cache
│   │   ├── builtin_types.py  # Built-in model types
│   │   ├── model_constructs.py # Deprecated aliases
│   │   └── executor.py       # Pure execution loop, LLM calls, validation hooks
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
