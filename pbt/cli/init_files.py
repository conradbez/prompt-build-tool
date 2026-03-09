"""
pbt/cli/init_files.py

Scaffold templates and the `pbt init` Click command.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Scaffold file templates
# ---------------------------------------------------------------------------

INIT_FILES = {
    "0-basic-usage.md": """\
# Getting Started

| Directory | Purpose |
|-----------|---------|
| `models/` | <- START HERE: Prompt files — see `models/0-basic-usage.md` |
| `tests/` | LLM-as-judge tests — see `tests/0-basic-usage.md` |
| `validation/` | Pre-pass quality gates — see `validation/0-basic-usage.md` |
| `outputs/` | Generated outputs from `pbt run` (auto-created) |

Run: `pbt run` or `pbt run --promptdata topic="your topic"`
""",
    "tests/0-basic-usage.md": """\
# Tests

Used when you change prompts to make sure everything still works and passes quality standards.
Write prompts here that will run against your models to assess their quality.

Each `.prompt` file in this directory is an LLM-as-judge test.
The prompt should reference model outputs via `{{ ref('model_name') }}` and return JSON:
  - Pass: `{"results": "pass"}`
  - Fail: `{"results": "fail"}`

Example test file `tests/summary_has_bullets.prompt`:

    Does the following text contain at least 3 bullet points (lines starting with - or •)?

    {{ ref('summary') }}

    Reply with only valid JSON: {"results": "pass"} or {"results": "fail"}.
""",
    "models/0-basic-usage.md": """\
# Models

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

## client.py
Add a `models/client.py` to configure which LLM to call. It must expose a
`llm_call(prompt: str) -> str` function. See the scaffolded example for a
Gemini implementation.

Run with: `pbt run` or `pbt run --promptdata topic="your topic"`
""",
    "validation/0-basic-usage.md": """\
# Validation

Optional Python code that runs *before* an LLM prompt output is passed to the next prompt.
Use this to check basic quality gates and avoid wasting tokens on bad intermediate results.

Each `.py` file must expose a `validate(prompt: str, result: str) -> bool` function.
Returning `False` stops the pipeline and reports a validation error.

Example `validation/article.py`:

    def validate(prompt: str, result: str) -> bool:
        \"\"\"Article must be at least 200 characters and contain a markdown header.\"\"\"
        return len(result) >= 200 and "#" in result
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
import os
from google import genai

def llm_call(prompt: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        contents=prompt,
    ).text
""",
    "openai": """\
import os
from openai import OpenAI

def llm_call(prompt: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
""",
    "anthropic": """\
import os
import anthropic

def llm_call(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=8096,
        messages=[{"role": "user", "content": prompt}],
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
        files["models/client.py"] = CLIENT_PY[provider.lower()]
        files["validation/articles.py"] = """\
import json
from pydantic import BaseModel, ValidationError


class Article(BaseModel):
    content: str
    author: str
    audience: str


def validate(prompt: str, result: str) -> bool:
    \"\"\"Article output must be valid JSON matching the Article model.\"\"\"
    try:
        data = json.loads(result)
        article = Article(**data)
    except (json.JSONDecodeError, ValidationError):
        return False
    return len(article.content) >= 200
"""
        files["validation/summaries.py"] = """\
import json
from pydantic import BaseModel, ValidationError


class SummaryItem(BaseModel):
    title: str
    summary: str
    key_points: list[str]


class Summaries(BaseModel):
    summaries: list[SummaryItem]


def validate(prompt: str, result: str) -> bool:
    \"\"\"Summaries output must be valid JSON matching the Summaries model.\"\"\"
    try:
        data = json.loads(result)
        summaries = Summaries(**data)
    except (json.JSONDecodeError, ValidationError):
        return False
    return len(summaries.summaries) >= 1 and len(summaries.summaries[0].key_points) >= 1
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
