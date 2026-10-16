# pbt — prompt-build-tool

A **data-enineering-inspired** prompt orchestration tool for LLMs.

LLMs lack human intuition and grounded reality checks, so pbt structures prompt engineering into a DAG of interconnected hypotheses (pbt models) paired with predictive reality checks (pbt tests) to keep development oriented without losing context or rewriting from scratch.

Write modular prompts in Jinja2, reference the output of other prompts with
`ref()`, and let **pbt** resolve dependencies. 

---


## Quick start

```bash
uv tool install prompt-build-tool --with anthropic && pbt init --provider anthropic
# pip install prompt-build-tool anthropic && pbt init --provider anthropic   # or --provider openai | gemini
export ANTHROPIC_API_KEY=your_key_here   # or OPENAI_API_KEY / GEMINI_API_KEY
```

Edit the `.prompt` files in `models/`, then:

```bash
pbt run
```

Optional extra model kinds (`python_modal`, `agent_modal`, `agent37`): `pip install "prompt-build-tool[extras]"`.

## Writing models

Use `ref('model_name')` to inject the output of another model:

```jinja
{# models/outline.prompt #}
Based on this topic, create a detailed outline:

{{ ref('topic') }}
```

---

## Tests (`pbt test`)

Run `tests/*.prompt` files against the latest run's outputs. Each test passes when the LLM returns `{"results": "pass"}`. YAML cases can instead use `expect` to compare selected output fields directly, without a judge or test prompt files; see [YAML expectations](docs/ADVANCED_USAGE.md#checking-expected-outputs-expect).

```bash
pbt test
```

## Passing files to models (`promptfiles`)

Models can receive files (PDFs, images, ...) alongside the prompt: declare them with `{{ config(promptfiles=["doc"]) }}`, then pass them with `pbt run --promptfile doc=report.pdf` or `pbt.run(..., promptfiles={"doc": "report.pdf"})`.
Your `llm_call` gets them as a `files` parameter when its signature declares one.
See [Passing files to models](docs/INTERMEDIATE_USAGE.md#passing-files-to-models-promptfiles) for multiple files, `config` and per-item files with `each=`.

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

## Python and agent models

Not every node needs an LLM call. `execute_python` runs the rendered template as Python, for deterministic steps like counting, parsing or reshaping:

```jinja
{# models/length.prompt #}
{{ config(model_type="execute_python") }}
output = len(ref('article').split())
```

The output is what the code prints, or else the `output` variable (`dict`/`list` become JSON). `ref()` works in the code and builds dependency edges as usual. The code runs in-process with no sandbox, so treat `.prompt` files as trusted.

`agent` hands the rendered template to a coding agent ([mini-swe-agent](https://mini-swe-agent.com), installed with pbt) as a task. The agent runs shell commands in `agent_dir` until it submits:

```jinja
{# models/fix_tests.prompt #}
{{ config(model_type="agent", agent_dir="./repo") }}
Make the failing test in tests/test_math.py pass.
```

- **Which LLM drives the agent:** set `agent_model = "anthropic/claude-sonnet-5"` (a [litellm](https://docs.litellm.ai/docs/providers) model name) in `client.py`.
- **Reading the result:** the output is a dict, so downstream use `{{ ref('fix_tests')['output'] }}` for what the agent submitted.
- **Caching:** an unchanged task is not re-run, same as Python models.

More (step and cost limits, files the agent writes, the opencode backend with MCP): [Agent models](docs/ADVANCED_USAGE.md#agent-models-model_typeagent).

---

## Intermediate usage

[docs/INTERMEDIATE_USAGE.md](docs/INTERMEDIATE_USAGE.md) covers:

- [Grouping models in folders](docs/INTERMEDIATE_USAGE.md#grouping-models-in-folders)
- [`pbt docs`](docs/INTERMEDIATE_USAGE.md#pbt-docs) — HTML report of every run
- [Passing variables with `promptdata()`](docs/INTERMEDIATE_USAGE.md#passing-variables-to-templates-promptdata)
- [Customising the LLM backend (`client.py`)](docs/INTERMEDIATE_USAGE.md#customising-the-llm-backend-clientpy)
- [Passing files to models](docs/INTERMEDIATE_USAGE.md#passing-files-to-models-promptfiles)
- [RAG inside prompts](docs/INTERMEDIATE_USAGE.md#rag-inside-prompts-ragpy)
- [Model kinds from installed packages](docs/INTERMEDIATE_USAGE.md#model-kinds-from-installed-packages)
- [Dynamically skipping a model](docs/INTERMEDIATE_USAGE.md#how-to-dynamically-skip-a-model)
- [Tests](docs/INTERMEDIATE_USAGE.md#pbt-tests) and [validation](docs/INTERMEDIATE_USAGE.md#validation-validation)

---

## Advanced usage

[docs/ADVANCED_USAGE.md](docs/ADVANCED_USAGE.md) covers:

- [Python API](docs/ADVANCED_USAGE.md#python-api), including live progress callbacks and per-run `secrets`
- [Token usage](docs/ADVANCED_USAGE.md#token-usage-pbtllmresult)
- [Bulk testing with YAML cases](docs/ADVANCED_USAGE.md#bulk-testing-with-yaml-cases-pbt-test)
- [`pbt ls`](docs/ADVANCED_USAGE.md#pbt-ls)
- [`pbt serve`](docs/ADVANCED_USAGE.md#pbt-serve) and the [HTTP server](docs/ADVANCED_USAGE.md#http-server-utilsserver)
- [Output format and `config()` keys](docs/ADVANCED_USAGE.md#output-format-config-config)
- [Global instructions](docs/ADVANCED_USAGE.md#global-instructions-globalprompt)
- [Map and reduce with `each=`](docs/ADVANCED_USAGE.md#map-and-reduce-each)
- [Models that produce files](docs/ADVANCED_USAGE.md#models-that-produce-files-pbtfile-pbtdir-pbtoutput)
- [Agent models](docs/ADVANCED_USAGE.md#agent-models-model_typeagent)
- [Writing your own model kind](docs/ADVANCED_USAGE.md#writing-your-own-model-kind-model_type)

Internals and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
