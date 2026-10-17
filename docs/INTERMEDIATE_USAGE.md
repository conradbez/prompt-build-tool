
### `promptmap docs`

Write an HTML report of every run: each model's prompt, output, timing and files, plus the DAG.

```bash
promptmap docs           # writes .promptmap/docs/index.html
promptmap docs --open    # and opens it
```

---

## Grouping models in folders

Group models in folders as the project grows. A leading `<number>_<letter>_`
only sorts files in reading order and is not part of the model name, so
`2_build/2_a_parts.prompt` is still `ref('parts')`:

```
map/
  1_design/
    1_a_brief.prompt
    1_b_requirements.prompt
  2_build/
    2_a_parts.prompt
  3_qa/
    3_a_bom_line.prompt
```

---

## Passing variables to templates (`promptdata()`)

Inject runtime variables into templates using the `promptdata("name")` function — similar to how dbt's `source()` and `ref()` work.

```bash
promptmap run --promptdata tone=formal --promptdata audience=engineers
```

```python
promptmap.run("map", promptdata={"tone": "formal", "audience": "engineers"})
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

## Project defaults (`promptdata.yml`, `promptfiles.yml`, `secrets.yml`)

Instead of repeating flags on every run, put defaults in flat `name: value` files in the project root (the directory you run `promptmap` from):

| File | Used by | Overridden by |
|------|---------|---------------|
| `promptdata.yml` | `promptmap run` | `--promptdata KEY=VALUE` |
| `promptfiles.yml` | `promptmap run` (paths relative to the project root) | `--promptfile NAME=PATH` |
| `secrets.yml` | `promptmap run`, `promptmap test`, `promptmap serve` | `--secret NAME=VALUE` |

CLI flags always win, key by key — so `promptdata.yml` can set `tone` and `topic` while `--promptdata topic=dogs` changes only `topic`. Keys with an empty value are ignored.

`promptmap init` doesn't create these files unless asked. Generate them, listing the keys your models use, in an existing project:

```bash
promptmap init --data      # promptdata.yml — every promptdata("...") key
promptmap init --files     # promptfiles.yml — every config(promptfiles=...) name
promptmap init --secrets   # secrets.yml + secrets-template.yml — secrets your model kinds require
```

or alongside a new project: `promptmap init my_project --secrets --data`.

Re-running keeps existing values and appends any new keys (`--force` rewrites the file). `--secrets` also adds `secrets.yml` to `.gitignore`: commit `secrets-template.yml` (same keys, no values) and keep real credentials out of git. Like `--secret`, values from `secrets.yml` are never rendered, cached or stored.

---

## Customising the LLM backend (`client.py`)

promptmap is unopinionated about which LLM you use. Create `client.py` at the project root (alongside your `map/` directory) and define an `llm_call` function — usually 5 lines:

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

promptmap will automatically discover and use this file. Run `promptmap init --provider <gemini|openai|anthropic|deepseek|qwen|kimi|xiaomi>` to scaffold a starter `client.py` for your chosen provider; add `--classifier` to also scaffold a jev-like `classify_call` for classifier-judged tests (P(yes) from one-token logprobs where the provider returns them — OpenAI, DeepSeek, Qwen — otherwise a hard yes/no). If the file exists but does not define `llm_call`, promptmap raises an error at startup.

`client.py` can also define `cache_check(prompt, previous) -> bool` to let a judge (e.g. that classifier) decide whether a prompt whose input data changed can reuse the last answer — see [`cache_check`](ADVANCED_USAGE.md#deciding-when-a-changed-prompt-is-worth-re-asking-cache_check).

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
promptmap run --promptfile my_document=report.pdf
promptmap run --promptfile report=annual.pdf --promptfile chart_image=q4.png
```

```python
promptmap.run("map", promptfiles={"my_document": "report.pdf"})
promptmap.run("map", promptfiles={"report": "annual.pdf", "chart_image": "q4.png"})
```

**3. Custom `llm_call` with file and config support:**

Accept optional `files` and/or `config` parameters in your `client.py` — promptmap passes them if the signature declares them:

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

## RAG inside prompts (`rag.py`)

`promptmap` has very little to say about RAG and leaves that up to you - you do this through the
`return_list_RAG_results(*args)` function `promptmap` give you access to in the .prompt template. `promptmap` will pass this call to
the `do_RAG` function you define in `rag.py` (at the project root, alongside your `map/` directory):

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
promptmap raises a clear error at render time.

---

## Model kinds from installed packages

A package can ship its own `model_type` values. Declare them under the
`promptmap.model_kinds` entry point group and `pip install` is all a user needs —
no import in `client.py`:

```toml
# the plugin's pyproject.toml
[project.entry-points."promptmap.model_kinds"]
my_kind = "my_pkg.my_kind"        # a module that calls @promptmap.model_kind on import
other   = "my_pkg:register"       # or a function promptmap calls once
```

promptmap loads them on the first `model_type` lookup. A plugin that fails to import
is logged and skipped. Set `PROMPTMAP_DISABLE_PLUGINS=1` to turn discovery off.

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

# promptmap tests


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

Add a `classify_call(state, question) -> float` to `client.py`. `promptmap.systemone_classifier()` builds one for any `/v1/systemone` API with no extra dependencies:

```python
import promptmap
classify_call = promptmap.systemone_classifier()  # hosted Jev; reads TYPESAFE_API_KEY
# classify_call = promptmap.systemone_classifier(model="laya:en", base_url="http://localhost:11435")  # Ollaya
test_judge = "classifier"                    # optional: make it the default judge
```

Runnable example with a local Ollaya (install, start, `promptmap test`): [`examples/classifier_test`](examples/classifier_test).

Use case — classifier tests on a chained pipeline: [`tests/README.md`](tests/README.md).

The judge is chosen per test with `{{ config(judge="llm") }}` / `{{ config(judge="classifier") }}`, otherwise by `promptmap test --judge llm|classifier`, otherwise by `test_judge` in `client.py`, otherwise `llm`. There is no fallback: a classifier test without a `---` line, or one that attaches `promptfiles`, is an error. LLM-judged tests are unchanged — the whole file, `---` included, goes to the LLM.

**Gate tests** — every test also guards `promptmap run` by default. `promptmap run` runs the models gate tests read (and everything upstream of them) first, then the gate tests, then the rest of the pipeline — only if every gate passed. A failing or erroring gate stops the run with exit code 1, and the downstream models are never called. With `--select`, a gate applies when all the models it reads are in the selected run, so `promptmap run -s article` still checks `outline` before `article`. `promptmap test` runs gate tests like any other test.

Opt a test out when it should only run under `promptmap test` — for example a subjective quality check that should not block a run, or one whose judge needs a service `promptmap run` should not depend on:

```jinja
{# tests/memo_reads_well.prompt #}
{# gate=False: style judgement; a borderline verdict should not block the run #}
{{ config(gate=False) }}
Is this memo clear and concise? {{ ref('memo') }}
Respond {"results": "pass"} or {"results": "fail"}.
```
 # Writing models


All standard Jinja2 syntax works too:

```jinja
{# map/comparison.prompt #}
{% set languages = ['Python', 'Go', 'Rust'] %}
Compare these languages for building CLI tools:
{% for lang in languages %}
- {{ lang }}
{% endfor %}

Context from previous analysis:
{{ ref('initial_analysis') }}
```

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

Run with `promptmap run` — validation fires automatically after each model's LLM call.
