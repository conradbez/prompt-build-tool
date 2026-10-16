# pbt — prompt-build-tool

A **data-engineering-inspired** prompt orchestration tool for LLMs (think `dbt` for prompts).

**Why `pbt`?** Treating complex LLM workflows as giant one-shot prompts is like navigating a city by staring at your feet—one wrong turn and you're lost. `pbt` turns your prompts into a visual map of connected hypotheses (`models`) and reality-check landmarks (`tests`), with built-in tools to inspect runs (`pbt docs`) and interactively edit the DAG (`pbt serve`).

**Mechanically, this means you:** write modular prompts in Jinja2, chain them into a DAG with `ref()`, and validate outputs against reality with `pbt test`. Iterate on complex LLM workflows without losing context or rewriting from scratch. *(Read more on the philosophy in [motivation.md](motivation.md)).*
![DAG visual](pbt_docs.png)

---

## Quick start

Install with `uv` (or `pip install prompt-build-tool anthropic`):

```bash
uv tool install prompt-build-tool --with anthropic
pbt init --provider anthropic   # or: openai | gemini
export ANTHROPIC_API_KEY=your_key_here

```

Edit your `.prompt` files in `models/`, then run the pipeline, test expectations, and inspect the run:

```bash
pbt run     # Execute the prompt DAG
pbt test    # Validate outputs against tests/
pbt docs    # Generate & open an HTML report of the run
pbt serve   # Launch the interactive local UI / HTTP server

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

*(Optional extra model kinds like `python_modal`, `agent_modal`, and `agent37` are available via `pip install "prompt-build-tool[extras]"`).*

### 3. Tests (`tests/*.prompt`)

Run `pbt test` to validate the latest run's outputs against your expectations:

* **LLM-as-a-judge:** Prompt tests in `tests/*.prompt` pass when the LLM returns `{"results": "pass"}`.
* **Deterministic checks:** Compare output fields directly in YAML using `expect` (no LLM judge needed).

### 4. Visualizing & serving (`pbt docs` & `pbt serve`)

Instead of digging through terminal logs, inspect and interact with your prompt graph visually:

* **`pbt docs`** generates a self-contained HTML report of every run—showing rendered prompts, model outputs, token usage, and test results across the DAG.
* **`pbt serve`** starts a local server to interactively explore the DAG, edit nodes, and trigger runs via HTTP or the visual UI.

---

## Documentation

| Category | Guides |
| --- | --- |
| **Inputs & Data** | [Passing variables (`promptdata`)](https://www.google.com/search?q=docs/INTERMEDIATE_USAGE.md%23passing-variables-to-templates-promptdata) · [Passing files (`promptfiles`)](https://www.google.com/search?q=docs/INTERMEDIATE_USAGE.md%23passing-files-to-models-promptfiles) · [RAG inside prompts](https://www.google.com/search?q=docs/INTERMEDIATE_USAGE.md%23rag-inside-prompts-ragpy) · [Global instructions](https://www.google.com/search?q=docs/ADVANCED_USAGE.md%23global-instructions-globalprompt) |
| **Workflow & DAG** | [Grouping models in folders](https://www.google.com/search?q=docs/INTERMEDIATE_USAGE.md%23grouping-models-in-folders) · [Map & reduce (`each=`)](https://www.google.com/search?q=docs/ADVANCED_USAGE.md%23map-and-reduce-each) · [Dynamically skipping models](https://www.google.com/search?q=docs/INTERMEDIATE_USAGE.md%23how-to-dynamically-skip-a-model) · [Models that produce files](https://www.google.com/search?q=docs/ADVANCED_USAGE.md%23models-that-produce-files-pbtfile-pbtdir-pbtoutput) |
| **Testing & Eval** | [Writing tests](https://www.google.com/search?q=docs/INTERMEDIATE_USAGE.md%23pbt-tests) · [Validation](https://www.google.com/search?q=docs/INTERMEDIATE_USAGE.md%23validation-validation) · [Bulk YAML testing & `expect](https://www.google.com/search?q=docs/ADVANCED_USAGE.md%23bulk-testing-with-yaml-cases-pbt-test)` |
| **CLI, UI & API** | [`pbt docs` (HTML report)](https://www.google.com/search?q=docs/INTERMEDIATE_USAGE.md%23pbt-docs) · [`pbt serve` & HTTP server](https://www.google.com/search?q=docs/ADVANCED_USAGE.md%23pbt-serve) · [`pbt ls`](https://www.google.com/search?q=docs/ADVANCED_USAGE.md%23pbt-ls) · [Python API & callbacks](https://www.google.com/search?q=docs/ADVANCED_USAGE.md%23python-api) · [Token usage](https://www.google.com/search?q=docs/ADVANCED_USAGE.md%23token-usage-pbtllmresult) |
| **Customization** | [Custom LLM backend (`client.py`)](https://www.google.com/search?q=docs/INTERMEDIATE_USAGE.md%23customising-the-llm-backend-clientpy) · [Output format & `config()](https://www.google.com/search?q=docs/ADVANCED_USAGE.md%23output-format-config-config)` · [Agent models](https://www.google.com/search?q=docs/ADVANCED_USAGE.md%23agent-models-model_typeagent) · [Custom model kinds](https://www.google.com/search?q=docs/ADVANCED_USAGE.md%23writing-your-own-model-kind-model_type) · [Architecture](https://www.google.com/search?q=docs/ARCHITECTURE.md) |
