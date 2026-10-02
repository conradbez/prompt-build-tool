# Advanced usage

Everything beyond the basics in the [README](../README.md).

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
    global_instruction="Answer in British English.",  # optional: text added to every prompt
)
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
| `global_instruction` | `str \| () -> str \| None` | Text rendered into every model's prompt. Falls back to `global.prompt` (next to models/) |
| `on_model_start` | `(name: str) -> None \| None` | Called just before each model starts |
| `on_model_done` | `(result: ModelRunResult) -> None \| None` | Called when each model finishes; `result.status` is `"success"`, `"error"` or `"skipped"` |

Returns a `dict` keyed by model name. Each value is the model's output string —
or `ModelStatus.SKIPPED` when an upstream model failed, or a `ModelError`
carrying the message when that model itself failed.

```python
results = pbt.run("models")

if isinstance(results["article"], pbt.ModelError):
    print("failed:", results["article"])
elif results["article"] is pbt.ModelStatus.SKIPPED:
    print("never ran")
else:
    print(results["article"])
```

`pbt.async_run(...)` takes the same arguments and returns the same dict, for
calling from inside an existing event loop.

### Live progress

Pass `on_model_start` / `on_model_done` to track every model's status while
the run is in progress, e.g. to drive a UI:

```python
statuses = {}
pbt.run(
    "models",
    on_model_start=lambda name: statuses.update({name: "running"}),
    on_model_done=lambda r: statuses.update({r.model_name: r.status}),
)
```

---

## Token usage (`pbt.LLMResult`)

Return `pbt.LLMResult(text, spent_tokens=...)` instead of the bare text and pbt records the tokens each model spent — `pbt docs` then shows, per run, tokens used, tokens served from cache, and an estimated cold-start total (used + cached). How you count `spent_tokens` is up to you; a sane default is input + output + thinking, or the provider's own total when it reports one. The scaffolded clients already do this; a plain string still works, tokens just show as `—`.

```python
import pbt

def llm_call(prompt: str) -> pbt.LLMResult:
    response = genai.Client().models.generate_content(...)
    return pbt.LLMResult(response.text, spent_tokens=response.usage_metadata.total_token_count)
```

---

## Bulk testing with YAML cases (`pbt test`)

List named sets of inputs in `promptparams.yml` or any `promptparams/*.yml` (all files are combined). `pbt test` runs the models once per case and reports each test as `test_name[case name]`. Shared inputs go in `baselines`; every case inherits `default` unless it `extends` another, and only lists what it changes:

```yaml
baselines:
  default:
    promptdata: {topic: The history of the printing press, audience: curious adults}
cases:
  - name: Default topic and audience
  - name: Rocket topic, default audience
    promptdata: {topic: How rockets reach orbit}
```

See [`examples/generate_articles_example/promptparams/`](../examples/generate_articles_example/promptparams/) for a full example, and the `pbt.promptparams` docstring for every rule. Handy flags: `--case "Rocket*"` to run matching cases, `--promptdata k=v` for a one-off case, `--save-case NAME` to keep it.

---

## `pbt serve`

Start the pbt HTTP server and open the docs page in the browser.

```bash
pbt serve
# pbt serve --host 0.0.0.0 --port 8000
```

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

## Output format config (`config()`)

Call `config()` at the top of a `.prompt` file to declare the expected output format:

```jinja
{{ config(output_format="json") }}
Return a JSON object with keys "title" and "summary".
```

When `output_format: json` is set, pbt validates the LLM output as JSON (stripping optional ` ```json ``` ` fences) and passes the parsed `dict`/`list` to downstream models via `ref()`, for example enabling `{{ ref('model').title }}` access.

### Recognised keys

| Key | Effect |
| --- | --- |
| `output_format` | `"json"` parses and validates the output as JSON; defaults to `"text"` |
| `output_extension` | File extension for `outputs/<model>.<ext>`; defaults to `"md"` |
| `promptfiles` | Names of files this model receives at runtime — see [Passing files to models](../README.md#passing-files-to-models-promptfiles) |
| `model_type` | `"template"`, `"execute_python"`, `"agent"`, or a type you register; defaults to a plain LLM call |
| `each` | Path to an upstream list; run this model once per item, any `model_type` — see [Map and reduce](#map-and-reduce-each) |
| `global_instruction` | `False` opts this model out of the run's [global instruction](#global-instructions-globalprompt) |

Any other key — and any unknown `model_type` — raises an `UnknownConfigKeyWarning` naming the model and file, with a did-you-mean suggestion, so typos like `output_fmt="json"` surface instead of being silently ignored. The key is still kept in the config dict, since pbt forwards the whole dict to a `llm_call(prompt, config=...)` that accepts one. If your `llm_call` consumes custom keys, register them once to silence the warning:

```python
import pbt

pbt.register_config_keys("temperature", "max_tokens")
```

---

## Global instructions (`global.prompt`)

Sometimes every prompt in a project needs the same preamble — a house style, a
persona, an output convention. This is pbt's analogue of dbt's `query-comment`:
one snippet, rendered into every prompt pbt sends.

Create a `global.prompt` next to your `models/` directory (the same place
`client.py` and `rag.py` live) and it is picked up automatically:

```
my_project/
├── client.py
├── global.prompt      ← rendered into every model's prompt
└── models/
    ├── article.prompt
    └── summary.prompt
```

```jinja
{# global.prompt #}
Write in British English. Never use em dashes.
```

By default the instruction is **prepended** to each model's prompt. Reference
`{{ prompt }}` to place the model body yourself:

```jinja
{# global.prompt — wrapper form #}
You are a careful technical writer.

<task>
{{ prompt }}
</task>

Answer in British English.
```

### It is a Jinja template too

The instruction is rendered with each model's own context, so
`{{ promptdata('key') }}`, `{{ model.name }}` and `{{ was_skipped('x') }}` all
work inside it:

```jinja
{% if promptdata("tone") %}Write in a {{ promptdata("tone") }} tone.{% endif %}
```

`ref()` is deliberately **not** available. The instruction goes into every
model, so a `ref('article')` would make every model depend on `article` —
including `article` itself. Use `promptdata()` for values that vary per run.

### Setting it from Python

```python
import pbt

pbt.run(global_instruction="Write in British English.")

# or build it at runtime — the callable is invoked once per run
pbt.run(global_instruction=lambda: load_house_style_from_somewhere())
```

The explicit argument wins over `global.prompt`. This is also the only way to
set one for `models_from_dict` runs, which never touch the filesystem. On the
CLI, `--global-instruction PATH` overrides the file for a single run:

```bash
pbt run --global-instruction experiments/terse.prompt
```

### Opting out

A single model opts out with `config()`:

```jinja
{{ config(global_instruction=False) }}
Return the raw JSON only, with no preamble.
```

Two exclusions are automatic:

- **`execute_python` models** never receive it — their template renders to
  Python source, and prepending prose to it would be a `SyntaxError`.
- **Test prompts** in `tests/` never receive it. The models under test render
  exactly as they do in a real run, but a judge told how to write is a biased
  judge.

Changing the instruction changes every rendered prompt, so the prompt cache
invalidates itself — the next `pbt run` re-runs affected models without
`--clear-cache`.

---

## Map and reduce (`each=`)

Set `each` in `config()` to run a model once per item of an upstream list, then collect the results back into a list. It works on any `model_type`.

| Model | Config | `ref()` inside yields | Output |
|---|---|---|---|
| `parts` | `output_format="json"` | | `{"parts": [p1, p2]}` |
| `check` | `each="parts.parts[*]"`, `model_type="execute_python"` | `ref('parts')` = `p1`, then `p2` | `[c1, c2]` |
| `chapters` | `output_format="json"` | | `[ch1, ch2]` |
| `sections` | `each="chapters"`, `output_format="json"` | `ref('chapters')` = `ch1`, then `ch2` | `[[s1, s2], [s3]]` |
| `polish` | `each="sections[*][*]"` | `ref('sections')` = `s1`, `s2`, `s3` | `[t1, t2, t3]` |
| `report` | none | `ref('polish')` = `[t1, t2, t3]` | one string |

```jinja
{# models/polish.prompt #}
{{ config(each="sections[*][*]") }}
Polish this section:
{{ ref('sections') }}
```

- **Map.** Inside an `each` model, `ref()` of the iterated model yields the current item. Every other `ref()` yields the full upstream value. The calls run concurrently and the output keeps input order.
- **Map over map.** Each `[*]` unnests one level, like Postgres `jsonb_array_elements`. `sections[*][*]` turns a list of lists into one flat list.
- **Reduce.** Any model that `ref()`s the mapped list. Use `model_type="template"` for a reduce with no LLM call.
- **Paths.** `.key`, `[n]` and `[*]`, in the SQL/JSON spelling. `each="chapters"` is shorthand for `chapters[*]`. A path that does not lead to a list fails with the type it found.
- **Per item rows and cache.** Each item is stored as its own row, `polish[0]`, `polish[1]`, … in `pbt docs` and the database. Its raw response sits under that item's own cache key, so on the next run only changed items reach the LLM.

The model named in `each` is a dependency even when the template never `ref()`s it. A skip function inside an `each` model applies to that item only; the model as a whole still succeeds.

Worked example: [`examples/mckinsey_hypothesis`](../examples/mckinsey_hypothesis), where one `branch` model tests every branch of a hypothesis tree and `synthesis` reduces them.

The same paths work in any `ref()`: `{{ ref('parts.parts[0].lcsc') }}`, or `{{ ref('sections[*][*]') }}` in a test to see exactly what `polish` iterated.

---

## Models that produce files (`pbt.File`, `pbt.Dir`, `pbt.Output`)

A model can output images, zips, PDFs or whole folders, with or without text,
and pass them to the models downstream of it. `llm_call` returns a file object
instead of a string, or a dict/list with file objects inside:

```python
# client.py
import pbt

def llm_call(prompt, files=None, config=None):
    ...
    return pbt.File(png_bytes, name="logo.png")                        # one file
    return pbt.Dir("build/site")                                       # a folder
    return pbt.Output("Here's the logo.", files=[pbt.File(png_bytes, name="logo.png")])  # text + files
    return {"caption": text, "image": pbt.File(png_bytes, name="logo.png")}              # anywhere in JSON
```

Downstream models use it in three ways:

```jinja
{# 1. As text: a one-line handle with the name, type, size and hash #}
Describe {{ ref('logo') }}          {# → [file: logo.png (image/png, 48.2 KB, sha256:ab12cd34ef56)] #}
{{ ref('logo_with_caption').caption }}

{# 2. Attached: the bytes go to llm_call(files=[...]) #}
{{ config(promptfiles=["logo"]) }}                 {# every file that model produced #}
{{ config(promptfiles=["logo_with_caption.image"]) }}   {# just one of them #}
```

Naming a model in `promptfiles` makes it a dependency, just like `ref()`. Any
other name is still a run-level `--promptfile`. Attached files are binary file
objects with `.name`, so the same `llm_call` code handles both kinds.

```python
# 3. In Python models: the objects themselves
{{ config(model_type="execute_python") }}
logo = ref('logo')
data = logo.read_bytes()          # also: logo.text, logo.open(), logo.path (a read-only copy on disk)
(out_dir / "thumb.png").write_bytes(make_thumbnail(data))   # files written to out_dir become the output
print("made a thumbnail")                                    # ...and printed text sits alongside them
```

`File`, `Dir` and `Output` are available in Python models without an import.

**Caching.** File bytes are stored once, by sha256, in the `blobs` table of
`.pbt/pbt.db`. The model's output keeps a small manifest of those hashes, so the
prompt cache serves files exactly as it serves text. A downstream model's cache
key covers the hashes of any files attached to it, so changed bytes mean a fresh
call. If a cached file's bytes have gone missing, pbt recomputes it.

**Where the files end up.** `pbt run` writes each model's files to
`outputs/<model>/`, next to its text in `outputs/<model>.md`. `pbt docs` shows
them in a gallery. `pbt show-result <model> --save-files DIR` exports any past
run's files. `pbt.run()` returns the `File`/`Dir`/`Output` objects.

**Keeping bytes elsewhere (e.g. S3).** A blob store is three methods (`put`,
`get` and `exists`), keyed by sha256. Define `blob_store` in `client.py`, as an
instance or a zero-argument function, and every command uses it:

```python
# client.py
import boto3

class S3BlobStore:
    def __init__(self, bucket, prefix="pbt/blobs/"):
        self.s3, self.bucket, self.prefix = boto3.client("s3"), bucket, prefix
    def put(self, sha256, data):
        if not self.exists(sha256):
            self.s3.put_object(Bucket=self.bucket, Key=self.prefix + sha256, Body=data)
    def get(self, sha256):
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self.prefix + sha256)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            raise KeyError(sha256) from None
    def exists(self, sha256):
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self.prefix + sha256)
            return True
        except Exception:
            return False

blob_store = S3BlobStore("my-bucket")
```

From Python, pass `pbt.run(..., blob_store=S3BlobStore("my-bucket"))` or
`SQLiteStorageBackend(blob_store=...)`.

**Safety.** Model output is never trusted to name a file. A stored file
reference is an object tagged `"$pbt"`, and only pbt writes those tags: the
same key in JSON a model returns is escaped, so it reads back as plain data.
Every reference is still checked when it is read back (hash format, file name,
relative paths, MIME type, the folder's tree hash), and every read checks the
bytes against their hash. `pbt.Dir` refuses symlinks, and file names can never
contain a path.

---

## Agent models (`model_type="agent"`)

An `agent` model hands its rendered template to a coding agent as a task. The
agent runs shell commands in `agent_dir` until it submits. Two backends:

| `agent_backend`  | what it is | install |
|------------------|------------|---------|
| `mini-swe-agent` (default) | [mini-swe-agent](https://mini-swe-agent.com): a minimal bash-only agent | `pip install "prompt-build-tool[agent]"` |
| `opencode`       | [opencode](https://opencode.ai): a full coding agent with file tools and **MCP servers** | `npm i -g opencode-ai` (or `curl -fsSL https://opencode.ai/install \| bash`) |

```bash
pip install "prompt-build-tool[agent]"              # mini-swe-agent
export MSWEA_MODEL_NAME=anthropic/claude-sonnet-5   # any litellm model name

npm i -g opencode-ai                                # opencode (npm, not pip)
opencode providers login                            # or export OPENAI_API_KEY etc.
```

```jinja
{# models/fix_tests.prompt #}
{{ config(model_type="agent", agent_dir="./repo", agent_step_limit="30") }}
Make the failing test in tests/test_math.py pass. Explain the fix in your final output.

{{ ref('bug_report') }}
```

The output is a dict:

| key        | value                                   |
|------------|-----------------------------------------|
| `output`   | what the agent submitted (its final reply, for opencode) |
| `logs`     | the full message trajectory (opencode's JSON events) |
| `time_run` | seconds the agent ran                   |

Downstream: `{{ ref('fix_tests')['output'] }}`.

| config key         | meaning                                      |
|--------------------|----------------------------------------------|
| `agent_dir`        | working directory (required; created if missing) |
| `agent_backend`    | `mini-swe-agent` (default) or `opencode`     |
| `agent_model`      | litellm name for mini-swe-agent (default `MSWEA_MODEL_NAME`); `provider/model` for opencode (default: opencode's own config) |
| `agent_step_limit` | max LLM calls, `0` = no limit (default `0`)  |
| `agent_cost_limit` | max spend in dollars, `0` = no limit (default `3`) |
| `agent_mcp`        | opencode only: MCP servers the agent may use (see below) |

Results are cached on the rendered prompt, so an unchanged task does not
re-run. Either agent runs commands on your machine with no sandbox (opencode
is started with `--auto`, so it approves its own tool calls).

### opencode with MCP servers (`agent_mcp`)

`agent_mcp` is opencode's [`mcp` config block](https://opencode.ai/docs/mcp-servers/):
a dict of server name to server config, either inline or as the path of an
`opencode.json` (relative to the `.prompt` file). pbt passes it to opencode
through `OPENCODE_CONFIG_CONTENT`, which opencode merges over its project and
global config, so an `opencode.json` already in `agent_dir` still applies.

```jinja
{# models/research.prompt #}
{{ config(
    model_type="agent",
    agent_backend="opencode",
    agent_dir="./scratch",
    agent_model="anthropic/claude-sonnet-4-5",
    agent_mcp={
        "docs": {"type": "remote", "url": "https://mcp.example.com/docs"},
        "fs":   {"type": "local", "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "./data"]},
    },
) }}
Use the docs MCP server to find how {{ ref('question') }} is configured, then
write the answer to answer.md and reply with its contents.
```

or, keeping the servers in a file: `agent_mcp="./opencode.json"`.

The `opencode` binary is found on `PATH`, or at `PBT_OPENCODE_BIN`. Step and
cost limits are enforced by pbt from opencode's per-step cost events: when
exceeded, the process is stopped and the last log entry is
`{"type": "exit", "reason": "..."}`.

---

## Writing your own model kind (`model_type=`)

Every `.prompt` file is run by a *model kind*. Leave `model_type` unset and pbt
sends the rendered prompt to your LLM; set it to `template`,
`execute_python` or `agent` and pbt runs it differently. If none of those do
what you need, you can add your own.

A kind is a function and a registration, both in `client.py`:

```python
# client.py
import pbt

@pbt.model_kind("shout", config_keys={"suffix"})
async def shout(rendered, call):
    response = await call.llm(rendered)
    return response.upper() + call.spec.config.get("suffix", "")
```

Then use it from any model:

```jinja
{# models/loud.prompt #}
{{ config(model_type="shout", suffix="!") }}
Summarise {{ ref('article') }}.
```

```bash
pbt run
```

### What you get

pbt renders the template for you and hands your function two things:

| | |
|---|---|
| `rendered` | The rendered prompt text, with every `ref()`, `promptdata()` and skip function already resolved |
| `call` | `call.llm(rendered)` sends it to your LLM; `call.spec` is the model being run (`call.spec.name`, `call.spec.config`); `call.outputs` holds upstream outputs by name; `call.compute(rendered, compute=fn)` caches arbitrary work |

Whatever you return becomes the model's output — the thing `ref('loud')` gives
downstream models, and the thing written to `outputs/`. Return a string and it
is parsed for you if the model sets `output_format="json"`. Return a list or a
dict and it is kept as-is. Return a `pbt.File`, `pbt.Dir` or `pbt.Output`, or a
list/dict with them inside, and the model produces
[files](#models-that-produce-files-pbtfile-pbtdir-pbtoutput): pbt stores the
bytes, caches them, and hands them downstream. See
[Produce or read files](#optional-extras) below.

`call.llm()` returns whatever your `llm_call` returns. That is usually a string,
but it can be a file object if your backend produces files. A kind that changes
the response, like `shout` above, should check the type before using string
methods on it.

Everything else keeps working without you doing anything: the prompt cache,
`{{ config(output_format="json") }}`, the skip functions, `validation/`,
`pbt test`, `pbt docs` and the run report all treat your kind like a built-in
one. `call.llm` is already cached, timed and skip-aware — there is nothing to
opt into.

`config_keys={"suffix"}` tells pbt which `config()` keys your kind reads.
Without it, `pbt run` warns that `suffix` looks like a typo.

### The full record

The decorator is shorthand for building a `ModelKind` and registering it. The
long form is what you want for a kind that has no `exec_fn` of its own:

```python
pbt.register_model_kind(pbt.ModelKind(
    name="shout",
    exec_fn=shout,
    config_keys={"suffix"},
))
```

| Field | Default | Meaning |
|---|---|---|
| `name` | — | The `config(model_type=...)` value |
| `exec_fn` | `None` | `async (rendered, call) -> Any`. `None` means the rendered text *is* the output |
| `config_keys` | `frozenset()` | The `config()` keys this kind reads |
| `accepts_global_instruction` | `True` | `False` when the rendered text is not a prompt for a model to answer |

The built-ins are nothing but this record. The default kind's `exec_fn` is
just `await call.llm(rendered)`; `template` is
`ModelKind("template", exec_fn=None, accepts_global_instruction=False)`.

### Optional extras

**Skip the LLM entirely.** Anything you can compute, you can return:

```python
@pbt.model_kind("truncate", config_keys={"max_words"})
async def truncate(rendered, call):
    return " ".join(rendered.split()[:call.spec.config_int("max_words", 50)])
```

```jinja
{{ config(model_type="truncate", max_words="30") }}
{{ ref('article') }}
```

The `ref()` calls in the template are what tell pbt your model has to run after
the models it references — reading `call.outputs` directly does not create that
edge.

For a *one-off* calculation, you do not need a kind at all:
[`execute_python`](../README.md#python-models-model_typeexecute_python) already runs a
model's template as Python. Write a kind when the behaviour is worth reusing
across models and configuring per model, the way `truncate` takes `max_words`.

**Fan out over a list.** Nothing to write: any model using your kind can set
`config(each="model.key[*]")`, and pbt renders once per item, runs your
`exec_fn` on each concurrently, and collects the results into a list.

**Cache expensive non-LLM work.** Anything slow and repeatable can go behind the
same cache your LLM calls use, so it does not re-run when nothing changed:

```python
return await call.compute(rendered, compute=lambda: scrape(rendered))
```

**Produce or read files.** Upstream files arrive in `call.outputs` as `pbt.File`,
`pbt.Dir` and `pbt.Output` objects, and returning one makes your model output
files. This kind zips every file its upstream models produced:

```python
import io, zipfile
import pbt

@pbt.model_kind("zip_files", config_keys={"zip_name"})
async def zip_files(rendered, call):
    def build():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name in call.spec.depends_on:
                value = call.outputs[name]
                files = value.files.values() if isinstance(value, pbt.Output) else [value]
                for f in files:
                    if isinstance(f, pbt.File):
                        # A fixed timestamp keeps the zip's bytes, and so its hash, stable.
                        info = zipfile.ZipInfo(f"{name}/{f.name}", date_time=(1980, 1, 1, 0, 0, 0))
                        zf.writestr(info, f.read_bytes())
        return pbt.File(buf.getvalue(), name=call.spec.config.get("zip_name", "bundle.zip"))

    return await call.compute(rendered, compute=build)
```

```jinja
{{ config(model_type="zip_files", zip_name="assets.zip") }}
Bundle {{ ref('logo') }} and {{ ref('diagram') }}
```

Three things make this work well:

- **`ref()` in the template does more than build the DAG edge.** A file renders
  as a handle that includes its hash, so `rendered` changes whenever an upstream
  file's bytes change. The cache key is built from `rendered`, so
  `call.compute(rendered, ...)` re-zips exactly when an input changed, and not
  otherwise.
- **Deterministic bytes.** A file's hash is its identity, so the same inputs
  should give the same bytes. Zips normally stamp the current time on each
  entry; the fixed `date_time` above prevents that. Without it, every re-zip
  would be a "new" file to downstream models.
- **No storage code.** pbt writes the bytes to the blob store, caches the
  manifest, shows the file in `pbt docs` and writes it to `outputs/<model>/`.
  The same happens for files returned from `call.llm()`, and for files a kind
  builds without `call.compute`.

A kind can also send an upstream model's files to the LLM without any code:
the model sets `{{ config(promptfiles=["logo"]) }}` and `call.llm()` attaches
them.

**Opt out of the global instruction.** If your rendered template is not a prompt
for a model to answer — Python source, or a value passed straight through — set
`accepts_global_instruction=False` so your
[global instruction](#global-instructions-globalprompt) is not prepended to it.

### Where to register it

Anywhere that runs before your models are read. `client.py` is the easiest spot,
because pbt already imports it on every `pbt run`, `pbt test` and `pbt ls`.

A worked example lives in [`examples/custom_model_type/`](../examples/custom_model_type/).

Upgrading from 0.3, where a custom type was a class? See
[docs/MIGRATION_model_kinds.md](MIGRATION_model_kinds.md).
