# pbt — Architecture & Developer Notes

---

## Call order

One `.prompt` file to one result, in the order the code actually runs:

```
1. pbt/__init__.py  ·  pbt/cli/          entry point
     └─ llm.py / rag.py                  resolve backends by importing client.py,
                                         which is also where custom kinds register

2. pbt/model_types.py                    the registry must be populated FIRST —
     └─ executor/builtin_kinds.py        build_spec asks it whether a model_type
                                         is known, and warns at parse time if not

3. pbt/executor/graph.py                 load_models → build_spec → expand → build_dag
     ├─ executor/parser_initial.py       static analysis of raw template text:
     │                                   config(), ref() deps, promptdata()
     ├─ pbt/model_spec.py                → one frozen ModelSpec per node
     └─ each kind's expand_fn            one node becomes several (quality_check)

4. pbt/executor/executor.py              walk the DAG in topological order
     ├─ executor/run_context.py          render, prompt cache, LLM call, timing
     │    └─ executor/parser_model.py    the Jinja pass: ref(), promptdata(),
     │                                   RAG, skip functions, global instruction
     └─ each kind's exec_fn              given (rendered, call) — the only step
                                         a model kind contributes

5. pbt/storage/                          run rows, outputs, prompt cache
     └─ sqlite.py | memory.py            (base.py is the interface)

   pbt/validator.py · tester.py · docs.py   validation, `pbt test`, `pbt docs`
```

Two things about that order are easy to get backwards:

**The registry comes before parsing, not before execution.** `graph.build_spec()`
calls `_resolve_model_type()`, which warns and falls back to a plain LLM call for
any `model_type` it does not recognise. So `client.py` must be imported before
`models/` is read, or a project-local kind looks like a typo and its models
silently run as ordinary prompts.

**There are two parsers, at two different times.** `parser_initial.py` is static
analysis over the raw text during graph construction — it never renders anything.
`parser_model.py` is the Jinja pass during execution, once per prompt. A change
that affects the DAG belongs in the first; a change that affects a prompt's
content belongs in the second.

`ModelSpec` has no position in this order: it is the data step 3 produces and
step 4 consumes, not a stage of its own.

### Dependency resolution

Step 3 scans every `*.prompt` for `ref('...')` with a regex, builds a directed
graph with [NetworkX](https://networkx.org/), and topologically sorts it for the
execution order. A cycle fails the run immediately with a clear error. At
execution time, a model whose dependency errored is marked **skipped** rather
than being sent to the LLM with a missing input.

---

## Key design decisions

**`executor.py` is a pure executor.** It takes callables (`llm_call`, `rag_call`, `validators`) and never touches the filesystem. File discovery lives in `pbt/cli/` and `pbt/__init__.py`. This keeps the executor testable with mock callables.

**Prompt cache is content-addressed.** The key is the *rendered* prompt (post-Jinja, pre-LLM), the model's `config()` block as sorted JSON, and a hash of any attached promptfiles, joined by NULs — `RunContext.cache_key()`. SQLite stores the SHA256 of that as `model_results.prompt_hash`. Config and files are part of the key because changing `output_format` or swapping an attached PDF must not serve the old answer.

**`--select` runs the full upstream chain fresh.** `pbt run --select tweet` runs `tweet` and all its ancestors in dependency order. The prompt cache makes unchanged upstream nodes instant — no stale-output risk, no need for a previous run of the same DAG.

**`model_outputs` is `dict[str, str | dict | list]`.** When a model declares `output_format: json`, its entry is a parsed Python object, not a string. Downstream `ref('model')` in Jinja receives the object, enabling `{{ ref('model').key }}` access. The DB always stores canonical JSON strings.

**Validators must not return the raw `result` string when `output_format: json` is set.** The executor parses the LLM output into a Python `dict`/`list` before running validators, storing it in `model_outputs`. If the validator returns the string `result` unchanged, it overwrites the parsed object with a string — breaking any downstream loop model that expects a list. Passthrough validators should either `return True` or return the already-parsed value (e.g. `json.loads(result)`). Validators that don't need to transform JSON output should simply be omitted.

**Model data, model behaviour, and run inputs are three separate things.** A `ModelSpec` (`pbt/model_spec.py`) is inert parsed data: source, deps, config, `model_type`. A `ModelKind` (`pbt/model_types.py`) is a frozen record describing how that data becomes an output — at most one function, `async exec_fn(rendered, call)`, plus four declarative fields. A `RunContext` (`pbt/executor/run_context.py`) carries every run-wide input — the backends, promptdata, promptfiles, validators, the global instruction, accumulated outputs — plus the shared mechanisms `render()`, `call_llm()` and `cached()`. Keeping them apart is why a new run-wide input is one field on `RunContext` rather than a new parameter on six functions, and why a new model kind is one function rather than an edit in the parser, the graph, and the executor.

**A kind is data, not a class.** A kind holds no per-model state, so there is nothing for an instance to carry and nothing for a base class to supply. `ModelKind` is a frozen dataclass; the registry is `dict[str, ModelKind]`. Three of the five built-ins therefore contribute no execution code at all: `template` is `exec_fn=None`, `loop` is the plain LLM call with `fan_out=True`, and `quality_check` is an `expand_fn`.

**A kind never sees the `RunContext`.** The executor renders, then binds the model and that render's state onto a `ModelCall` (`spec`, `outputs`, `llm`, `compute`) and hands it to `exec_fn` alongside the rendered text. `call.llm(rendered)` and `call.compute(rendered, compute=fn)` are `functools.partial`s over `RunContext.call_llm` / `RunContext.cached`, already cached, timed and skip-aware. So a kind receives what it needs pre-loaded instead of a god object it must navigate, and caching stays outside the kind entirely.

**`config()` keys are validated, not enforced.** `known_config_keys()` in `parser_initial.py` unions three sources: `_BUILTIN_CONFIG_KEYS` (the keys pbt acts on for every model), the `config_keys` each registered model kind declares, and anything passed to `pbt.register_config_keys(...)` for a custom `llm_call`. A key outside all three (or an unregistered `model_type`) triggers an `UnknownConfigKeyWarning` at load time naming the model and file. It stays a warning because the whole config dict is forwarded to any `llm_call` accepting a `config` parameter, so custom keys are legitimate. **A kind-specific config key belongs on that kind's `config_keys`**, not in `_BUILTIN_CONFIG_KEYS` — registering the kind is then the only step needed.

**`ModelKind.expand_fn` enables DAG-build-time node expansion.** After each model is parsed, `graph.py` calls its kind's `expand_fn(spec, all_specs)`, if it has one. Returning a list of specs replaces that one node with all of them, before the DAG is built; returning `None` leaves it alone. The list must contain exactly one node keeping the declared name, so downstream `ref()` calls still resolve — `graph.py` raises if it does not. `all_specs` is read-only, for looking up already-parsed upstream models. `quality_check` uses this to expand one node into an interleaved chain of check and retry nodes, terminating in a `template` node that carries the original name.

**The executor owns the lifecycle; a kind owns one step.** `execute_model()` in `executor.py` resolves the kind, renders, asks its `exec_fn` for a value, then applies skip propagation, `output_format` parsing, storage, and validation — identically for built-in and user-registered kinds. `RunContext.cached()` is the single caching path: `call_llm` is a thin wrapper over it, and `execute_python` reaches it as `call.compute` so unchanged code does not re-execute.

**Fan-out is the executor's job, not a kind's.** `fan_out=True` tells `execute_model()` to resolve the single upstream dependency whose output is a JSON list, render once per item (`primary=False`, so one skipped item does not mark the whole model skipped), run `exec_fn` on each concurrently via `asyncio.gather`, and collect the results in input order. `loop` is therefore the ordinary LLM call with one flag set, and any registered kind can fan out.

**Storage keeps raw and validated output separately.** `mark_model_success` stores the raw output, which is what the prompt cache serves, so editing a validator never forces a new LLM call. When a validator transforms the output, the executor also calls `record_validated_output`, and `get_model_outputs_from_run` prefers it — so `pbt test` judges the value the pipeline actually passed downstream.

---

## Adding a model kind

One function and one registration. Everything else — rendering, caching,
skipping, JSON parsing, validation, storage — is the executor's job and already
applies.

```python
import pbt

@pbt.model_kind("shout", config_keys={"suffix"})
async def shout(rendered, call):
    response = await call.llm(rendered)
    return response.upper() + call.spec.config.get("suffix", "")
```

```jinja
{# models/loud.prompt #}
{{ config(model_type="shout", suffix="!") }}
Summarise {{ ref('article') }}.
```

The decorator returns the function unchanged, so it stays directly callable and
testable. The equivalent explicit form, which is also how you register a kind
that has no `exec_fn` of its own:

```python
pbt.register_model_kind(pbt.ModelKind("shout", shout, config_keys={"suffix"}))
```

`ModelKind` fields:

| Field | Meaning |
|---|---|
| `name` | The `config(model_type=...)` value. `""` is the plain LLM call |
| `exec_fn` | `async (rendered, call) -> Any`. `None` means the rendered text *is* the output — no LLM call |
| `fan_out` | Render once per item of an upstream JSON list and run `exec_fn` on each concurrently |
| `expand_fn` | `(spec, all_specs) -> list[ModelSpec] \| None` — rewrite this node into several at DAG-build time |
| `config_keys` | The `config()` keys this kind consumes, so they stop warning |
| `accepts_global_instruction` | `False` when the rendered template is not a natural-language prompt |

`call` is a `ModelCall`: `call.spec` (the model, for its name and `config()`),
`call.outputs` (upstream outputs by name), `call.llm(rendered)` and
`call.compute(rendered, compute=fn)` — both already cached, timed and
skip-aware.

Put the registration in `client.py`: pbt imports it before parsing any
`.prompt` file, precisely so project-local kinds are available. Registering
anywhere that runs earlier works too.

Return value of `exec_fn`: a `str` is parsed when the model sets
`output_format="json"`; anything else (a list, a dict) is stored as-is.

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
- Three tables: `runs`, `model_results`, `test_results`.
- `prompt_hash` is indexed (with `completed_at DESC`) for cache lookups.


### `runs`

One row per `pbt run` invocation.

| Column | Type | Description |
|---|---|---|
| `run_id` | TEXT PK | UUID for the run |
| `run_date` | TEXT | Calendar date of the run |
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
| `prompt_hash` | TEXT | SHA256 of the cache key — the prompt cache's index |
| `llm_output` | TEXT | Raw LLM response text (what the prompt cache serves) |
| `llm_output_validated` | TEXT | Post-validation output, when a validator changed it |
| `cached` | INTEGER | 1 when every call this model made was a cache hit |
| `started_at` | TIMESTAMP | Execution start |
| `completed_at` | TIMESTAMP | Execution end |
| `execution_ms` | INTEGER | Wall-clock time in milliseconds |
| `error` | TEXT | Error message if status = `error` |
| `depends_on` | TEXT | JSON list of upstream model names |
| `model_type` | TEXT | The model's kind; `''` for a plain LLM call |
| `config` | TEXT | The model's `config()` block as JSON |

Query results directly:

```bash
sqlite3 .pbt/pbt.db "SELECT model_name, status, execution_ms FROM model_results ORDER BY id DESC LIMIT 10"
```

---

## Python API

`pbt.run()` is the sync entry point; `pbt.async_run()` is the same function
without the `asyncio.run()` wrapper, for calling from inside an event loop. The
[README](../README.md#python-api) documents every parameter — the notes here are
the ones that matter when working on pbt itself.

```python
import pbt

results = pbt.run("path/to/models")

for name, output in results.items():
    print(name, output)
```

**Both return a `dict`, not a list.** Keyed by model name, the value is the
model's output string — or `ModelStatus.SKIPPED` if an upstream model failed, or
a `ModelError` carrying the message if this one did. `ModelRunResult` (with
`prompt_rendered`, `execution_ms`, `cached`, …) is what `execute_run()` returns
internally; the public API reduces it to the value a caller wants. Read the rest
back out of storage, or out of `pbt docs`.

Three parameters exist for pbt's own tests and tooling rather than for everyday
use:

| Parameter | Type | Purpose |
|---|---|---|
| `models_from_dict` | `dict[str, str] \| None` | Models as `{name: template_source}`, bypassing the filesystem entirely |
| `storage_backend` | `StorageBackend \| None` | Swap SQLite for `MemoryStorageBackend` (or your own). Defaults to `.pbt/pbt.db` |
| `verbose` | `bool` | The dbt-style progress log. `False` silences it |

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

What a pbt *project* looks like — the package's own layout is the call order above.

```
my-project/
├── client.py            # optional: llm_call(prompt) + custom model kinds
├── rag.py               # optional: do_RAG(...)
├── global.prompt        # optional: instruction prepended to every prompt
├── models/
│   ├── topic.prompt     # no dependencies
│   ├── outline.prompt   # depends on topic
│   └── article.prompt   # depends on topic + outline
├── validation/          # optional: per-model validate(prompt, result) -> bool
├── outputs/             # written by `pbt run`
└── .pbt/
    ├── pbt.db           # runs, model results, test results, prompt cache
    └── docs/index.html  # written by `pbt docs`
```

---
