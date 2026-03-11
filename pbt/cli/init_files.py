"""
pbt/cli/init_files.py

Scaffold templates and the `pbt init` Click command.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Scaffold file templates
# ---------------------------------------------------------------------------

INIT_FILES = {
    "README.md": """\
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
        \"\"\"Article must be at least 200 characters and contain a markdown header.\"\"\"
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
""",
    "models/articles.prompt": """\
{{ config(output_format="json") }}
{# Generate an article on a topic. Pass --promptdata topic=... or leave blank for a generic article. #}
{% if promptdata("topic") %}
Write a detailed, engaging article about: {{ promptdata("topic") }}
{% else %}
Write a detailed, engaging article about a fascinating topic of your choice.
{% endif %}

Structure it with an introduction, 3-4 body sections, and a conclusion.
Use markdown formatting with headers.

Return only valid JSON in this exact format:
{
  "content": "<full article text in markdown>",
  "author": "<a plausible author name>",
  "audience": "<intended audience, e.g. 'general public', 'developers', 'students'>"
}
""",
    "models/summaries.prompt": """\
{{ config(output_format="json") }}
You are given a JSON article object. Produce a concise summary.

Article:
{{ ref('articles') }}

Return only valid JSON in this exact format:
{
  "summaries": [
    {
      "title": "<short title for the summary>",
      "summary": "<2-3 sentence summary of the article>",
      "key_points": ["<point 1>", "<point 2>", "<point 3>"]
    }
  ]
}
""",
    "tests/summary_has_bullets.prompt": """\
Does the following JSON summaries object contain at least 3 key points in the first summary entry?

{{ ref('summaries') }}

Reply with only valid JSON: {"results": "pass"} or {"results": "fail"}.
""",
}

CLIENT_PY: dict[str, str] = {
    "gemini": """\
import mimetypes
import os
from google import genai

# Automatically picked up by `pbt` to run .prompt files.
# Optional kwargs passed by pbt when declared in the signature:
#   files  - list of open file objects attached to the prompt (via --promptdata key=@path)
#   config - dict of {{ config(...) }} options from the .prompt file (e.g. {"output_format": "json"})
def llm_call(prompt: str, files: list | None = None, config: dict | None = None) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    uploaded = []
    for f in (files or []):
        mime, _ = mimetypes.guess_type(f.name)
        uploaded.append(client.files.upload(
            file=f,
            config={"mime_type": mime or "text/plain"},
        ))
    contents = [prompt] + uploaded if uploaded else prompt
    return client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        contents=contents,
    ).text
""",
    "openai": """\
import os
from openai import OpenAI

# Automatically picked up by `pbt` to run .prompt files.
# Optional kwargs passed by pbt when declared in the signature:
#   files  - list of open file objects attached to the prompt (via --promptdata key=@path)
#   config - dict of {{ config(...) }} options from the .prompt file (e.g. {"output_format": "json"})
def llm_call(prompt: str, files: list | None = None, config: dict | None = None) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    content: list = [{"type": "input_text", "text": prompt}]
    for f in (files or []):
        uploaded = client.files.create(file=f, purpose="user_data")
        content.append({"type": "input_file", "file_id": uploaded.id})
    response = client.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        input=[{"role": "user", "content": content}],
    )
    return response.output_text
""",
    "anthropic": """\
import os
import anthropic

# Automatically picked up by `pbt` to run .prompt files.
# Optional kwargs passed by pbt when declared in the signature:
#   files  - list of open file objects attached to the prompt (via --promptdata key=@path)
#   config - dict of {{ config(...) }} options from the .prompt file (e.g. {"output_format": "json"})
def llm_call(prompt: str, files: list | None = None, config: dict | None = None) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    content: list = []
    for f in (files or []):
        uploaded = client.beta.files.upload(file=f)
        content.append({"type": "document", "source": {"type": "file_id", "value": uploaded.id}})
    content.append({"type": "text", "text": prompt})
    message = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=8096,
        messages=[{"role": "user", "content": content}],
    )
    return message.content[0].text
""",
}

PROVIDERS = ("gemini", "openai", "anthropic")


# ---------------------------------------------------------------------------
# CLI command — registered onto the main Click group from cli/__init__.py
# ---------------------------------------------------------------------------

def register_command(main) -> None:
    """Attach the `pbt init` command to *main* Click group."""
    import click
    from rich.console import Console
    from pbt.cli.vscode import is_running_in_vscode, setup_vscode_associations

    console = Console()

    @main.command("init")
    @click.argument("project_name", default="generate_articles_example")
    @click.option("--force", is_flag=True, default=False, help="Overwrite existing files.")
    @click.option(
        "--provider",
        type=click.Choice(PROVIDERS, case_sensitive=False),
        default="gemini",
        show_default=True,
        help="LLM provider to use in the generated client.py.",
    )
    def init(project_name: str, force: bool, provider: str) -> None:
        """Scaffold a starter pbt project inside PROJECT_NAME/."""
        files = dict(INIT_FILES)
        files["client.py"] = CLIENT_PY[provider.lower()]
        files["validation/articles.py"] = """\
# Validation files let you post-process and gate LLM outputs before they flow downstream.
# - Name this file after the prompt it validates (e.g. articles.py validates articles.prompt).
# - Return False (or raise) to fail the pipeline; return any value to replace the raw LLM text.
# - Returned values are stored and used by ref() in downstream prompts — parse, clean, or reshape freely.

def validate(prompt: str, result: str):
    \"\"\"Passthrough — replace this with your own logic.\"\"\"
    return result


# Example: parse and validate a JSON article with Pydantic
#
# import json
# from pydantic import BaseModel, ValidationError
#
# class Article(BaseModel):
#     content: str
#     author: str
#     audience: str
#
# def validate(prompt: str, result: str) -> Article | bool:
#     try:
#         data = json.loads(result)
#         article = Article(**data)
#     except (json.JSONDecodeError, ValidationError):
#         return False
#     if len(article.content) < 200:
#         return False
#     return article.model_dump()
"""

        root = Path(project_name)
        created, skipped = [], []

        for rel_path, content in files.items():
            path = root / rel_path
            if path.exists() and not force:
                skipped.append(str(path))
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            created.append(str(path))

        for f in created:
            console.print(f"  [green]created[/green]  {f}")
        for f in skipped:
            console.print(f"  [dim]skipped[/dim]  {f}  [dim](use --force to overwrite)[/dim]")

        if created:
            console.print(f"\nRun [bold cyan]cd {project_name} && pbt run[/bold cyan], or [bold cyan]pbt run --promptdata topic='your topic'[/bold cyan]")

        if is_running_in_vscode():
            setup_vscode_associations()
            console.print("  [dim]VS Code detected — configured .vscode/settings.json for .prompt files[/dim]")
