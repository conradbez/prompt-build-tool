# pbt — prompt-build-tool

A **data-enineering-inspired** prompt orchestration tool for LLMs.

Write modular prompts in Jinja2, reference the output of other prompts with
`ref()`, and let **pbt** resolve dependencies. 

---


## Quick start

### 1. Install

```bash
pip install prompt-build-tool

# Also install the SDK for your LLM provider:
# pip install google-genai      # Gemini
# pip install openai            # OpenAI
# pip install anthropic         # Anthropic
```

### 2. Generate example

```bash
pbt init --provider anthropic
# pbt init --provider openai
# pbt init --provider gemini
```

### 3. Set your API key

```bash
export ANTHROPIC_API_KEY=your_key_here
# export OPENAI_API_KEY=your_key_here
# export GEMINI_API_KEY=your_key_here
```

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
| `global.prompt` | `query-comment` in `dbt_project.yml` |
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

**Attaching model files (`promptfiles`)** — a test attaches files the models produced with the same `config()` syntax models use, so the judge sees the actual bytes rather than a `[file: …]` handle. Name a model for all its files, or `model.key` for one:

```jinja
{# tests/logo_is_a_fox.prompt #}
{{ config(promptfiles=["logo"]) }}
Is the attached image a fox? Respond {"results": "pass"} or {"results": "fail"}.
```

Names that aren't models are run-level `--promptfile`s. A test that declares no `promptfiles` gets every run-level promptfile, as before. The attached bytes are part of the test's cache key.

**Test cases (`promptparams`)** — to test many sets of inputs, describe them as named cases in YAML. `pbt test` runs the models once per case and reports every test against each one, as `test_name[case name]`.

Every `*.yml`/`*.yaml` file in `promptparams/`, plus `promptparams.yml`, is loaded and combined, so you can split cases by topic and keep shared inputs in one place:

```yaml
# promptparams/base.yml — shared inputs
baselines:
  default:                          # every case starts from this...
    promptdata:
      tone: formal
      audience: engineers
    promptfiles:
      document: reports/annual.pdf  # relative to this file
  casual:
    extends: default                # baselines can build on each other
    promptdata:
      tone: casual
```

```yaml
# promptparams/tone.yml — only what each case changes
cases:
  - name: Formal report for engineers         # ...unless it says otherwise
  - name: Casual tone for developers
    extends: casual
    promptdata:
      audience: developers
  - name: Summary without the source document
    promptfiles:
      document: null                          # null removes an inherited input
  - name: Stands on its own
    extends: []                               # skip the default baseline
    promptdata:
      tone: terse
      audience: children
```

- `name` is how the case is reported and must be unique. A case without one is named `<file>_<n>`.
- `extends` takes a baseline name or a list of them, applied left to right with the case's own values on top. With no `extends`, a case inherits `default` if there is one.
- `promptdata` values can be any YAML value, multi-line strings included. `promptfiles` are a path or a list of paths, relative to the file that declares them.
- Unknown keys, unknown baselines, loops and duplicate names are errors, so typos fail loudly.

```bash
pbt test                                   # every case
pbt test --case "Casual*"                  # cases whose name matches (glob, case-insensitive)
pbt test --promptparams regressions/       # other files or directories (repeatable)
pbt test --check-latest                    # ignore cases; test the latest stored run
```

`pbt test` also writes `promptparams.yml.example`, a starter file listing every input the models and tests use.

**Inline params (`--promptdata` / `--promptfile`)** — pass params straight to `pbt test` to run the models with them and test against that run. They form a single case that inherits the `default` baseline like any other, or the baselines you name with `--extends`:

```bash
pbt test --promptdata tone=playful
pbt test --promptdata audience=kids --extends casual
```

**Save a run as a case (`--save-case`)** — when an inline param set is worth keeping as a regression case, name it. pbt writes it to its own file in `promptparams/` (`promptparams/playful_tone.yml`), so it is re-tested in future runs. Only the inline values are saved, so the case keeps inheriting its baselines:

```bash
pbt test --promptdata tone=playful --save-case "Playful tone"
```

> **Moving from `promptparams.csv`:** CSV is no longer read. Each row becomes a case: `promptdata.tone` → `promptdata: {tone: …}` and `promptfile.doc` → `promptfiles: {doc: …}`. Inputs shared by most rows can go in a `default` baseline. `--add-to-csv` is now `--save-case NAME`.


### `pbt serve`

Start the pbt HTTP server and open the docs page in the browser.

```bash
pbt serve
# pbt serve --host 0.0.0.0 --port 8000
```


### `pbt docs`

Generate an HTML report of all previous runs with expandable model details and a DAG diagram.
Files the models produced get a gallery for the latest run, plus a preview and
download link next to each model's output. They are copied into
`.pbt/docs/files/` beside the report, so the folder can be opened, zipped or
served as it is. `pbt serve` serves them too.

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

## Customising the LLM backend (`client.py`)

pbt is unopinionated about which LLM you use. Create `client.py` at the project root (alongside your `models/` directory) and define an `llm_call` function — usually 5 lines:

```python
# client.py (Anthropic example)
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

pbt will automatically discover and use this file. Run `pbt init --provider <anthropic|openai|gemini>` to scaffold a starter `client.py` for your chosen provider. If the file exists but does not define `llm_call`, pbt raises an error at startup.

---

## RAG inside prompts (`rag.py`)

`pbt` has very little to say about RAG and leaves that up to you - you do this through the
`return_list_RAG_results(*args)` function `pbt` give you access to in the .prompt template. `pbt` will pass this call to
the `do_RAG` function you define in `rag.py` (at the project root, alongside your `models/` directory):

```python
# rag.py
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

If `rag.py` is absent and a template calls `return_list_RAG_results`,
pbt raises a clear error at render time.

---


## Passing files to models (`promptfiles`)

Models can receive files (PDFs, images, etc.) alongside the text prompt. Declare the files a model needs via `config()`, then provide the actual paths at runtime.

**1. Declare in config:**

```jinja
{{ config(promptfiles=["my_document"]) }}
Summarise the attached document in 3 bullet points.
```

Multiple files use a JSON array:

```jinja
{{ config(promptfiles=["report", "chart_image"]) }}
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

Accept optional `files` and/or `config` parameters in your `client.py` — pbt passes them if the signature declares them:

```python
# client.py
def llm_call(prompt: str, files: list[str] | None = None, config: dict | None = None) -> str:
    # files  — resolved file paths declared via config(promptfiles=...)
    # config — the full config dict for this model, e.g. {"output_format": "json"}
    ...
```

Both parameters are optional and independent — declare either, both, or neither.

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
| `promptfiles` | Names of files this model receives at runtime — see [Passing files to models](#passing-files-to-models-promptfiles) |
| `model_type` | `"template"`, `"execute_python"`, or a type you register; defaults to a plain LLM call |
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

## Template models (`model_type="template"`)

A `template` model renders its Jinja and uses the result as its output, with no
LLM call — for nodes that only reshape what upstream models already produced.

```jinja
{# models/report.prompt #}
{{ config(model_type="template") }}

# {{ ref('title') }}

{{ ref('summary') }}
```

---

## Python models (`model_type="execute_python"`)

An `execute_python` model runs its template as Python instead of sending it to
the LLM. Use it for the deterministic steps in a pipeline — counting, parsing,
reshaping, arithmetic on an upstream result.

```jinja
{# models/length.prompt #}
{{ config(model_type="execute_python") }}
output = len(ref('article').split())
```

The template is rendered first, then the result is executed. Inside the code,
`ref('name')` returns an upstream model's output and `model_outputs` holds them
all. Those `ref()` calls are also what build the dependency edges, exactly as in
a normal prompt.

The output is whatever the code prints; if it prints nothing, a variable named
`output` is used instead (`dict` and `list` are JSON-encoded, anything else via
`str()`). Printing wins over `output`.

Results are cached on the rendered code, so unchanged code does not re-run. The
code executes in-process with full builtins and no sandbox, so treat a `.prompt`
file as trusted code.

---

## Agent models (`model_type="agent"`)

An `agent` model hands its rendered template to
[mini-swe-agent](https://mini-swe-agent.com) as a task. The agent runs shell
commands in `agent_dir` until it submits.

```bash
pip install "prompt-build-tool[agent]"
export MSWEA_MODEL_NAME=anthropic/claude-sonnet-5   # any litellm model name
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
| `output`   | what the agent submitted                |
| `logs`     | the full message trajectory             |
| `time_run` | seconds the agent ran                   |

Downstream: `{{ ref('fix_tests')['output'] }}`.

| config key         | meaning                                      |
|--------------------|----------------------------------------------|
| `agent_dir`        | working directory (required; created if missing) |
| `agent_model`      | litellm model name; default `MSWEA_MODEL_NAME` |
| `agent_step_limit` | max LLM calls, `0` = no limit (default `0`)  |
| `agent_cost_limit` | max spend in dollars, `0` = no limit (default `3`) |

Results are cached on the rendered prompt, so an unchanged task does not
re-run. The agent runs commands on your machine with no sandbox.

---

## Validation (`validation/`)

Create a `validation/` directory with Python files matching model names. Each file must define `validate(prompt, result) -> bool`. If it returns `False`, the model is marked as an error and stops it use in downstream models.

```python
# validation/article.py
import json
from pydantic import BaseModel, ValidationError


class Article(BaseModel):
    content: str
    author: str
    audience: str


def validate(prompt: str, result: str) -> bool:
    """Article output must be valid JSON matching the Article model."""
    try:
        data = json.loads(result)
        article = Article(**data)
    except (json.JSONDecodeError, ValidationError):
        return False
    return len(article.content) >= 200
```

Run with `pbt run` — validation fires automatically after each model's LLM call.

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

Use `{{ skip_and_set_to_value("value") }}` to skip the LLM call during Jinja rendering and provide the output directly:

```jinja
{% if "no action needed" in ref('previous_model') %}
{{ skip_and_set_to_value("No action needed.") }}
{% else %}
Summarise the following: {{ ref('previous_model') }}
{% endif %}
```

The model is recorded as a successful run, downstream templates can detect it with `was_skipped('model_name')`, and downstream `ref()` calls receive the value you provided.

---

## Writing your own model kind (`model_type=`)

Every `.prompt` file is run by a *model kind*. Leave `model_type` unset and pbt
sends the rendered prompt to your LLM; set it to `template` or
`execute_python` and pbt runs it differently. If neither does what you need,
you can add your own.

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
[`execute_python`](#python-models-model_typeexecute_python) already runs a
model's template as Python. Write a kind when the behaviour is worth reusing
across models and configuring per model, the way `truncate` takes `max_words`.

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

A worked example lives in [`examples/custom_model_type/`](examples/custom_model_type/).

Upgrading from 0.3, where a custom type was a class? See
[docs/MIGRATION_model_kinds.md](docs/MIGRATION_model_kinds.md).
