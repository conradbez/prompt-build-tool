# Getting Started

| File / Directory | Purpose |
|-----------------|---------|
| `client.py` | LLM backend (which model/API to call) |
| `models/` | START HERE: Prompt files |
| `tests/` | LLM-as-judge tests |
| `validation/` | Pre-pass quality gates |
| `outputs/` | Generated outputs from `pbt run` (auto-created) |

Run: `pbt run` or `pbt run --promptdata topic="your topic"`

---

## models/

Write your prompts here. Each `.prompt` file defines one step in your pipeline.

You can:
- Reference other prompt outputs:  `{{ ref('other_prompt_name') }}`
- Access passed-in data:           `{{ promptdata("key") }}`
- Configure model behaviour:       `{{ config(output_format="json") }}`

Example chain — `models/topic.prompt` → `models/article.prompt` → `models/summary.prompt`:

    # topic.prompt
    Generate a catchy blog post topic about AI.

    # article.prompt
    Write a detailed article about: {{ ref('topic') }}

    # summary.prompt
    {{ config(output_format="json") }}
    Summarise this article in 3 bullet points. Return JSON: {"bullets": ["...", "...", "..."]}.
    {{ ref('article') }}

---

## tests/

Used when you change prompts to make sure everything still works and passes quality standards.

Each `.prompt` file in this directory is an LLM-as-judge test.
The prompt should reference model outputs via `{{ ref('model_name') }}` and return JSON:
  - Pass: `{"results": "pass"}`
  - Fail: `{"results": "fail"}`

Example `tests/summary_has_bullets.prompt`:

    Does the following text contain at least 3 bullet points (lines starting with - or •)?

    {{ ref('summary') }}

    Reply with only valid JSON: {"results": "pass"} or {"results": "fail"}.

---

## validation/

Optional Python code that post-processes each model's LLM output before it is
stored and passed to downstream models via `ref()`.

Each `.py` file must expose a `validate(prompt: str, result: str) -> Any` function.
- Return `False` (or raise) to stop the pipeline and report a validation error.
- Return any other value to use it as the model's output — this replaces the raw
  LLM text for both storage and downstream `ref()` calls.

Example `validation/article.py` — fail short outputs, pass the rest unchanged:

    def validate(prompt: str, result: str) -> Any:
        """Article must be at least 200 characters and contain a markdown header."""
        if len(result) < 200 or "#" not in result:
            return False
        return result

Example with post-processing — parse and return a cleaned dict:

    import json

    def validate(prompt: str, result: str) -> Any:
        data = json.loads(result)          # raises → model fails
        data.pop("debug_info", None)       # strip internal keys
        return data                        # dict replaces raw JSON string downstream

---

## client.py

The `client.py` at the project root configures which LLM to call. It must expose a
`llm_call(prompt: str) -> str` function. See the scaffolded example for details.
