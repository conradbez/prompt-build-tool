# promptmap

A **data-engineering-inspired** prompt orchestration tool for LLMs (think `dbt` for prompts).

**Why `promptmap`?** Treating complex LLM workflows as giant one-shot prompts is like navigating a city by staring at your feet, one wrong turn and you're lost. `promptmap` turns your prompts into a visual map of connected hypotheses (`models`) and reality-check landmarks (`tests`), with built-in tools to inspect runs (`promptmap docs`).

**Mechanically, this means you:** write modular prompts in Jinja2, chain them into a dependency graph with `ref()`, and validate outputs against reality with `promptmap test`. Iterate on complex LLM workflows without losing context or rewriting from scratch. *(Read more on the philosophy in [motivation.md](motivation.md)).*

![DAG visual](promptmap_docs.png)

---

## Quick start

```bash
uv tool install promptmap --with anthropic  # or: openai | gemini
promptmap init --provider anthropic   # or: openai | gemini

```

Edit your `.prompt` files in `models/`, then run the pipeline, test expectations, and inspect the run:

```bash
export ANTHROPIC_API_KEY=your_key_here

promptmap run     # Execute the prompt DAG
promptmap test    # Validate outputs against tests/
promptmap docs    # Generate HTML report of the run and run history
promptmap serve   # Launch the interactive local UI / HTTP server
```

---

## Core concepts

### 1. Models (`models/*.prompt`)

Each `.prompt` file is a node in your DAG. Use `ref('model_name')` to inject the output of an upstream model:

```jinja
{# models/outline.prompt #}
Based on this topic, create a detailed outline:

{{ ref('topic') }}

```

### 2. Model types (`config(model_type=...)`)

Not every node in your pipeline needs to be a standard LLM call. Override `model_type` at the top of a `.prompt` file for deterministic steps, formatting, or coding agents:

| Model Type | What it does | Declaration |
| --- | --- | --- |
| **LLM** *(default)* | Sends the rendered Jinja prompt to your configured LLM. | Standard `.prompt` file |
| **`template`** | Renders Jinja directly with no LLM call (for reshaping upstream outputs). | `{{ config(model_type="template") }}` |
| **`execute_python`** | Runs rendered template as Python; captures `stdout` or `output` variable. | `{{ config(model_type="execute_python") }}` |
| **`agent`** | Hands the rendered prompt as a task to a coding agent (`mini-swe-agent`). | `{{ config(model_type="agent", agent_dir="./repo") }}` |

*(Optional extra model kinds like `python_modal`, `agent_modal`, and `agent37` are available via `pip install "promptmap[extras]"`).*

### 3. Tests (`tests/*.prompt`)

Run `promptmap test` to validate the latest run's outputs against your expectations:

* **LLM-as-a-judge:** Prompt tests in `tests/*.prompt` pass when the LLM returns `{"results": "pass"}`.
* **Deterministic checks:** Compare output fields directly in YAML using `expect` (no LLM judge needed).

### 4. Visualizing & serving (`promptmap docs` & `promptmap serve`)

Inspect prompt graph visually:

* **`promptmap docs`** generates a self-contained HTML report of every run—showing rendered prompts, model outputs, token usage, and test results across the DAG.
* **`promptmap serve`** starts a local server to interactively explore the DAG, trigger runs or inspect previous results.

---

## Documentation

| Category | Guides |
| --- | --- |
| **Inputs & Data** | [Passing variables (`promptdata`)](docs/INTERMEDIATE_USAGE.md#passing-variables-to-templates-promptdata) · [Passing files (`promptfiles`)](docs/INTERMEDIATE_USAGE.md#passing-files-to-models-promptfiles) · [RAG inside prompts](docs/INTERMEDIATE_USAGE.md#rag-inside-prompts-ragpy) · [Global instructions](docs/ADVANCED_USAGE.md#global-instructions-globalprompt) |
| **Workflow & DAG** | [Grouping models in folders](docs/INTERMEDIATE_USAGE.md#grouping-models-in-folders) · [Map & reduce (`each=`)](docs/ADVANCED_USAGE.md#map-and-reduce-each) · [Dynamically skipping models](docs/INTERMEDIATE_USAGE.md#how-to-dynamically-skip-a-model) · [Models that produce files](docs/ADVANCED_USAGE.md#models-that-produce-files-promptmapfile-promptmapdir-promptmapoutput) |
| **Testing & Eval** | [Writing tests](docs/INTERMEDIATE_USAGE.md#promptmap-tests) · [Validation](docs/INTERMEDIATE_USAGE.md#validation-validation) · [Bulk YAML testing & `expect`](docs/ADVANCED_USAGE.md#bulk-testing-with-yaml-cases-promptmap-test) |
| **CLI, UI & API** | [`promptmap docs` (HTML report)](docs/INTERMEDIATE_USAGE.md#promptmap-docs) · [`promptmap serve` & HTTP server](docs/ADVANCED_USAGE.md#promptmap-serve) · [`promptmap ls`](docs/ADVANCED_USAGE.md#promptmap-ls) · [Python API & callbacks](docs/ADVANCED_USAGE.md#python-api) · [Token usage](docs/ADVANCED_USAGE.md#token-usage-promptmapllmresult) |
| **Customization** | [Custom LLM backend (`client.py`)](docs/INTERMEDIATE_USAGE.md#customising-the-llm-backend-clientpy) · [Output format & `config()`](docs/ADVANCED_USAGE.md#output-format-config-config) · [Agent models](docs/ADVANCED_USAGE.md#agent-models-model_typeagent) · [Custom model kinds](docs/ADVANCED_USAGE.md#writing-your-own-model-kind-model_type) · [Architecture](docs/ARCHITECTURE.md) |
