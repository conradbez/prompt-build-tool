# pbt — Architecture & Developer Notes

Developer reference for the internals of prompt-build-tool. README covers user-facing features; this document covers *why* things are designed the way they are.

---

## Module map

```
pbt/
  __init__.py   Python API entry point (pbt.run)
  cli.py        Click commands — orchestrates discovery, calls execute_run
  graph.py      PromptModel dataclass, DAG building, topological sort
  parser.py     Jinja2 rendering, VarSpy, config block parsing
  executor.py   Pure execution loop — no file discovery, no CLI concerns
  llm.py        Resolves LLM backend (client.py → Gemini fallback)
  rag.py        Resolves RAG backend (rag.py → stub fallback)
  db.py         SQLite schema + all query helpers
  tester.py     Test runner (tests/*.prompt)
  validator.py  Validation framework (validation/*.py)
  docs.py       HTML report generator
```

---

## Execution flow

```
pbt run / pbt.run()
  │
  ├─ load_models(models_dir)          # graph.py: scan *.prompt, parse deps + config
  ├─ execution_order(models)          # graph.py: topological sort (lexicographic)
  ├─ compute_dag_hash(models)         # graph.py: SHA256 of names+edges only
  ├─ resolve_llm_call(models_dir)     # llm.py: client.py or Gemini
  ├─ resolve_rag_call(models_dir)     # rag.py: rag.py::do_RAG or stub
  ├─ load_validators(validation_dir)  # validator.py: validation/*.py
  ├─ db.create_run(...)               # SQLite: INSERT into runs
  │
  └─ execute_run(...)                 # executor.py
       for each model (topo order):
         ├─ skip if upstream failed
         ├─ render_prompt(source, model_outputs, vars, rag_call)  # Jinja2
         ├─ check SKIP_SENTINEL
         ├─ check prompt cache (SHA256 lookup in DB)
         ├─ llm_call(rendered)
         ├─ validate output_format: json  (if configured)
         ├─ run_validator(model, prompt, result)
         └─ db.mark_model_success(...)
```

---

## Key design decisions

### executor.py is a pure executor

`execute_run()` takes callables (`llm_call`, `rag_call`, `validators`) — it does **not** discover files or resolve backends. That responsibility sits in `cli.py` (for the CLI path) and `pbt/__init__.py` (for the Python API path). This keeps `executor.py` testable in isolation with mock callables.

### DAG hash covers structure, not content

`compute_dag_hash()` hashes only model names and dependency edges — not prompt content. This is intentional: the hash is used to verify that a previous run's outputs are safe to reuse for `--select`. If content changes but structure doesn't, `--select` still works correctly (the stale content warning is the user's responsibility).

### Prompt cache is content-addressed

`db.get_cached_llm_output()` uses SHA256 of the **rendered** prompt (post-Jinja, pre-LLM). Identical rendered prompts across any run return the cached output instantly. This is separate from `--select` (which reuses by run_id) — the cache operates at the rendered-string level across all history.

### model_outputs is typed `dict[str, str | dict | list]`

After `output_format: json` parsing, a model's entry in `model_outputs` is a Python `dict`/`list`, not a string. Downstream `ref('model')` in Jinja receives the parsed object, enabling `{{ ref('model').key }}` access. The DB always stores canonical JSON strings regardless.

---

## VarSpy: static var detection

Templates use `{{ vars.key }}` to access user-provided variables. Before any run, pbt does a **dry render** of every template to discover which vars it uses.

**How it works** (`parser.py`):

```python
class VarSpy(dict):
    def __getitem__(self, key):
        self._accessed.append(key)
        return f"__var_{key}__"   # truthy dummy — keeps rendering going

    def __contains__(self, key):
        return True               # vars.key always "exists"

    def get(self, key, default=None):
        self._accessed.append(key)
        return f"__var_{key}__"
```

`detect_used_vars(template_source)` renders the template with `VarSpy` as `vars`, a `defaultdict` for `ref()` calls, and dummy stubs for RAG/skip. Errors are swallowed — keys accessed up to the error are still returned.

**Stored on `PromptModel.vars_used`** after `load_models()`.

**Shown in `pbt ls`** (Vars used column) and **warned about in `pbt run`** if used vars are not provided via `--var`.

**Known limitation**: only one branch of a conditional is traversed. If a template has `{% if vars.flag %}{{ vars.a }}{% else %}{{ vars.b }}{% endif %}`, only `flag` and `a` are detected (the `if` branch is always taken because `VarSpy` returns a truthy dummy). Vars in `else`/`elif` branches that depend on other vars may be missed.

---

## Config block parsing

`.prompt` files can declare a config block as a Jinja comment — this means it is silently ignored by Jinja rendering and only parsed statically:

```
{# pbt:config
output_format: json
#}
```

`parse_model_config()` in `parser.py` extracts the block with a regex and parses `KEY: VALUE` lines. Stored in `PromptModel.config`. Currently supported keys:

| Key | Values | Effect |
|---|---|---|
| `output_format` | `text` (default), `json` | If `json`: strips ` ```json ``` ` fences, parses with `json.loads`, raises `ValueError` on failure, stores dict in `model_outputs` |

---

## Validation vs. tests

| | `tests/` | `validation/` |
|---|---|---|
| **Format** | `.prompt` Jinja files | `.py` Python files |
| **When runs** | Explicit `pbt test` command after a run | Automatically after each model in `pbt run` |
| **Input** | Model outputs from a previous run | The rendered prompt + LLM output of the just-executed model |
| **Pass criterion** | LLM returns `{"results": "pass"}` | `validate(prompt, result) -> bool` returns `True` |
| **Failure effect** | Non-zero exit from `pbt test` | Model marked as `error`, downstream models skipped |

---

## Backend resolution order

### LLM (`llm.py`)

1. Caller-provided `llm_call` argument (Python API or CLI-level override)
2. `{models_dir}/client.py` — must define `llm_call(prompt: str) -> str`
3. `client.py` one directory above `models_dir`
4. Built-in Gemini (`google-genai`, requires `GEMINI_API_KEY`)

### RAG (`rag.py`)

1. Caller-provided `rag_call` argument
2. `{models_dir}/rag.py` — must define `do_RAG(*args) -> list[str] | str`
3. `rag.py` one directory above `models_dir`
4. Stub that raises `RuntimeError` if a template actually calls `return_list_RAG_results()`

---

## SQLite notes

- DB lives at `.pbt/pbt.db` (relative to cwd at invocation time).
- `PRAGMA journal_mode=WAL` — allows concurrent readers during a run.
- `_migrate()` in `db.py` applies idempotent `ALTER TABLE ADD COLUMN` migrations for backward compatibility.
- `prompt_hash` (SHA256 of rendered prompt) is the cache key — indexed for fast lookup.
- `dag_hash` on the `runs` table is indexed for fast `--select` lookup.

---

## Adding a new CLI command

1. Add a function decorated with `@main.command("name")` in `cli.py`.
2. Import any new `pbt.*` modules at the top of `cli.py`.
3. If the command needs a new module, create `pbt/yourmodule.py`.
4. Update `ARCHITECTURE.md` and `README.md`.

## Adding a new config key

1. Add parsing logic in `parse_model_config()` (`parser.py`) if needed.
2. Handle the key in `executor.py` (post-LLM output processing).
3. Document in the Config block table above and in `README.md`.
