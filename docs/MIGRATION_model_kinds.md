# Migration: 0.3 → 0.4 — model types became model kinds

In 0.3 a custom `model_type` was a class subclassing `pbt.BaseModelType`. In 0.4
it is a `ModelKind` record holding at most one function. Nothing about `.prompt`
files changed — `{{ config(model_type="shout") }}` still selects your kind by the
same name, and every project without a custom type upgrades with no edits at
all.

Only code that **registers a custom type** needs changing, which in practice
means `client.py`.

```toml
# pyproject.toml
prompt-build-tool>=0.4
```

## The move

```python
# 0.3
@pbt.model_type("shout", config_keys={"suffix"})
class Shout(pbt.BaseModelType):
    async def execute(self, spec, ctx):
        rendered, state = ctx.render(spec)
        output = await ctx.call_llm(rendered, spec, state)
        return output.upper() + spec.config.get("suffix", "")

# 0.4
@pbt.model_kind("shout", config_keys={"suffix"})
async def shout(rendered, call):
    response = await call.llm(rendered)
    return response.upper() + call.spec.config.get("suffix", "")
```

Three things happen in that diff:

1. The class becomes the `execute` body as a plain function. There is no base
   class to subclass and no instance to construct.
2. **pbt renders for you.** `ctx.render(spec)` is gone from your code — the
   executor renders before calling you and passes the text as `rendered`.
3. `ctx` becomes `call`, a small record with the model and this render's state
   already bound, so nothing has to be threaded through by hand.

## Name-by-name

| 0.3 | 0.4 |
|---|---|
| `@pbt.model_type(name, config_keys=…)` on a class | `@pbt.model_kind(name, config_keys=…)` on an `async def` |
| `pbt.register_model_type(name, Cls)` | `pbt.register_model_kind(pbt.ModelKind(name, exec_fn, …))` |
| `pbt.BaseModelType` | *(gone — a kind is data, not a class)* |
| `pbt.ModelType` | `pbt.ModelKind` |
| `async def execute(self, spec, ctx)` | `async def exec_fn(rendered, call)` |
| `rendered, state = ctx.render(spec)` | *(gone — `rendered` is your first argument)* |
| `await ctx.call_llm(rendered, spec, state)` | `await call.llm(rendered)` |
| `await ctx.cached(rendered, spec, state, compute=fn)` | `await call.compute(rendered, compute=fn)` |
| `spec.config` / `spec.name` | `call.spec.config` / `call.spec.name` |
| `ctx.outputs` | `call.outputs` |
| `accepts_global_instruction = False` (class attribute) | `accepts_global_instruction=False` (kind field) |
| `def expand(self, spec, all_specs)` | *(removed — kinds can no longer rewrite the DAG)* |
| `pbt.get_model_type` / `pbt.known_model_types` | `pbt.get_model_kind` / `pbt.known_model_kinds` |
| `pbt.executor.builtin_types` | `pbt.executor.builtin_kinds` |
| `pbt.executor.model_constructs` (`BaseModelHandler`, …) | *(deleted — it aliased the pre-registry classes)* |

## Cases that get shorter

**A type that never called the LLM** — one that only reshaped the rendered text
— is now `exec_fn=None`, and you delete the function entirely:

```python
pbt.register_model_kind(pbt.ModelKind("passthrough", exec_fn=None))
```

That is all the built-in `template` kind is.

## Removed after 0.4

The `loop` and `quality_check` built-in kinds are gone, along with the
`ModelKind.fan_out` and `ModelKind.expand_fn` hooks that existed only to support
them, the `loop_over` / `quality_retries` / `quality_pass_marker` config keys,
and `model.meta` in templates. A `.prompt` file still using
`model_type="loop"` or `"quality_check"` now warns about an unknown
`model_type` and runs as a plain LLM call. A kind that needs one call per item
can loop inside its own `exec_fn` over `call.outputs`.

## What did not change in 0.4

- `.prompt` files, including the `model_type=` config key and the built-in
  names of the time (`template`, `execute_python`, and the since-removed
  `loop` and `quality_check`).
- Stored runs. `.pbt/pbt.db` and the prompt cache carry over — a 0.3 cache still
  serves 0.4 runs, because the cache key formula is unchanged.
- Where you register: `client.py`, or anywhere that runs before models are read.
- `config_keys` — still how you stop pbt reporting your options as typos.
- Everything the executor already did for you: the prompt cache,
  `output_format="json"` parsing, skip functions, validation, storage,
  `pbt test`, `pbt docs`. `call.llm` and `call.compute` are cached, timed and
  skip-aware exactly as `ctx.call_llm` and `ctx.cached` were.

See [ARCHITECTURE.md](ARCHITECTURE.md#adding-a-model-kind) for the full
`ModelKind` field reference, and
[`examples/custom_model_type/`](../examples/custom_model_type/) for a worked
example.
