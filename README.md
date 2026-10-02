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

# Optional: coding agents for model_type="agent"
# pip install "prompt-build-tool[agent]"   # mini-swe-agent
# npm i -g opencode-ai                     # opencode, with MCP servers
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

Group models in folders as the project grows. A leading `<number>_<letter>_`
only sorts files in reading order and is not part of the model name, so
`2_build/2_a_parts.prompt` is still `ref('parts')`:

```
models/
  1_design/
    1_a_brief.prompt
    1_b_requirements.prompt
  2_build/
    2_a_parts.prompt
  3_qa/
    3_a_bom_line.prompt
```

Model names must be unique across all folders.

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
| `ref('model.key[0]')` | `ref()` plus a JSON path, like `col -> 'key' -> 0` |
| `config(each='model.key[*]')` | `json_each` / `jsonb_array_elements`: one row per item |
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

**Classifier judge** — instead of an LLM, a test can be judged by a classifier such as [Jev](https://typesafe.ai) or a local [Ollaya](https://ollaya.dev) model, which answers a yes/no question with a probability. Put the question above a `---` line and the text to judge below it; the test passes when P(yes) ≥ 0.5 (or `{{ config(threshold=0.8) }}`):

```jinja
{# tests/haiku_has_three_lines.prompt #}
Does this haiku have exactly three lines?
---
{{ ref('haiku') }}
```

Add a `classify_call(state, question) -> float` to `client.py`. `pbt.systemone_classifier()` builds one for any `/v1/systemone` API with no extra dependencies:

```python
import pbt
classify_call = pbt.systemone_classifier()  # hosted Jev; reads TYPESAFE_API_KEY
# classify_call = pbt.systemone_classifier(model="laya:en", base_url="http://localhost:11435")  # Ollaya
test_judge = "classifier"                    # optional: make it the default judge
```

Runnable example with a local Ollaya (install, start, `pbt test`): [`examples/classifier_test`](examples/classifier_test).

Use case — classifier tests on a chained pipeline: [`tests/README.md`](tests/README.md).

The judge is chosen per test with `{{ config(judge="llm") }}` / `{{ config(judge="classifier") }}`, otherwise by `pbt test --judge llm|classifier`, otherwise by `test_judge` in `client.py`, otherwise `llm`. There is no fallback: a classifier test without a `---` line, or one that attaches `promptfiles`, is an error. LLM-judged tests are unchanged — the whole file, `---` included, goes to the LLM.

### `pbt docs`

Write an HTML report of every run: each model's prompt, output, timing and files, plus the DAG.

```bash
pbt docs           # writes .pbt/docs/index.html
pbt docs --open    # and opens it
```

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

pbt will automatically discover and use this file. Run `pbt init --provider <gemini|openai|anthropic|deepseek|qwen|kimi|xiaomi>` to scaffold a starter `client.py` for your chosen provider; add `--classifier` to also scaffold a jev-like `classify_call` for classifier-judged tests (P(yes) from one-token logprobs where the provider returns them — OpenAI, DeepSeek, Qwen — otherwise a hard yes/no). If the file exists but does not define `llm_call`, pbt raises an error at startup.

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

A model with `{{ config(each="imgs") }}` can iterate over a list of files, or over the files of a `Dir`;
`{{ config(promptfiles=[...]) }}` then attaches each item's own file.

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

## Advanced usage

[docs/ADVANCED_USAGE.md](docs/ADVANCED_USAGE.md) covers:

- [Python API](docs/ADVANCED_USAGE.md#python-api), including live progress callbacks
- [Token usage](docs/ADVANCED_USAGE.md#token-usage-pbtllmresult)
- [Bulk testing with YAML cases](docs/ADVANCED_USAGE.md#bulk-testing-with-yaml-cases-pbt-test)
- [`pbt serve`](docs/ADVANCED_USAGE.md#pbt-serve) and the [HTTP server](docs/ADVANCED_USAGE.md#http-server-utilsserver)
- [Output format and `config()` keys](docs/ADVANCED_USAGE.md#output-format-config-config)
- [Global instructions](docs/ADVANCED_USAGE.md#global-instructions-globalprompt)
- [Map and reduce with `each=`](docs/ADVANCED_USAGE.md#map-and-reduce-each)
- [Models that produce files](docs/ADVANCED_USAGE.md#models-that-produce-files-pbtfile-pbtdir-pbtoutput)
- [Agent models](docs/ADVANCED_USAGE.md#agent-models-model_typeagent)
- [Writing your own model kind](docs/ADVANCED_USAGE.md#writing-your-own-model-kind-model_type)

Internals and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
