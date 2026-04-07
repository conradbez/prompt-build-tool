# execute_python Model Type

Instead of sending a prompt to an LLM, it renders the `.prompt` file as a Jinja2 template, then runs the result as Python code via `exec()`.

## Execution Flow

1. **Render** — The `.prompt` file is rendered as a Jinja2 template. Upstream model outputs are available via `ref('model_name')` and the `model_outputs` dict.

2. **Execute** — The rendered string is compiled and `exec()`'d. Two variables are injected into the local namespace:
   - `model_outputs` — dict of all upstream outputs
   - `ref` — callable: `ref('name')` → `model_outputs.get('name')`

3. **Capture output** — priority order:
   - **stdout first**: anything `print()`'d is captured (trailing newline stripped)
   - **`output` variable fallback**: if stdout is empty and a variable named `output` exists in the local scope, it's used (`dict`/`list` → JSON-serialized, else `str()`)
   - **empty string**: if neither exists

4. **Post-process** — if `output_format='json'` is set in config, the output is parsed as JSON (strips ` ```json ``` ` fences if present); the parsed object is what downstream `ref()` calls see.

5. **Cache** — the result is stored keyed on the rendered code + config hash, so identical code doesn't re-execute.

## Final Output (`ModelRunResult.llm_output`)

| Scenario | `llm_output` value |
|---|---|
| `print("hello")` | `"hello"` |
| `output = {"key": 1}` (no print) | `'{"key": 1}'` |
| `output_format='json'` | JSON string of the parsed result |
| Skip triggered | The skip value string |
| Cached | Previously stored output |

Downstream models reference the output via `ref('this_model_name')` — which returns the parsed object (`dict`/`list`) if JSON format, or the raw string otherwise.
