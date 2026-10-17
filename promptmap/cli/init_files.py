"""
promptmap/cli/init_files.py

Scaffold templates and the `promptmap init` Click command.
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
| `outputs/` | Generated outputs from `promptmap run` (auto-created) |

Run: `promptmap run` or `promptmap run --promptdata topic="your topic"`

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

Tests are also gates: `promptmap run` judges each one right after the models it
reads, and stops the run if it fails. Add `{{ config(gate=False) }}` to a
test that should only run under `promptmap test`.

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

Scaffolded with `promptmap init --classifier`, it also has `classifier_for_test_feedback`,
which judges `{{ config(judge="classifier") }}` tests with a one-token yes/no call.
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
    "models/README.md": """\
# promptmap models (The Map)

Rather than treating prompts as isolated pass/fail one-shots — or saving an
unorganized mess of every prompt ever tried — promptmap models structures prompts as
a Directed Acyclic Graph (DAG) of lower- and higher-order hypotheses. Like
keeping failed working in physics, it externalizes your mental map so you can
tweak underlying assumptions rather than doing blind, one-shot rewrites.
""",
    "tests/README.md": """\
# promptmap tests (The Landmarks)

Tests preserve hard-earned domain lessons across iterations (such as catching
a reversed diode where logical pin names matched, but physical pad connections
were backward). They act as smoke tests to verify that a change actually moved
the project toward the destination.
""",
    "tests/summary_has_bullets.prompt": """\
{# gate=False: summaries is the last model; nothing downstream for a gate to protect #}
{{ config(gate=False) }}
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

import promptmap

# Automatically picked up by `promptmap` to run .prompt files.
# Optional kwargs passed by promptmap when declared in the signature:
#   files  - list of open file objects attached to the prompt (via --promptdata key=@path)
#   config - dict of {{ config(...) }} options from the .prompt file (e.g. {"output_format": "json"})
def llm_call(prompt: str, files: list | None = None, config: dict | None = None) -> promptmap.LLMResult:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    uploaded = []
    for f in (files or []):
        mime, _ = mimetypes.guess_type(f.name)
        uploaded.append(client.files.upload(
            file=f,
            config={"mime_type": mime or "text/plain"},
        ))
    contents = [prompt] + uploaded if uploaded else prompt
    response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        contents=contents,
    )
    # promptmap.LLMResult reports tokens spent to `promptmap docs`; returning the bare text also works.
    # total_token_count covers input, output and thinking.
    usage = response.usage_metadata
    return promptmap.LLMResult(response.text, spent_tokens=usage.total_token_count if usage else None)
""",
    "openai": """\
import os
from openai import OpenAI

import promptmap

# Automatically picked up by `promptmap` to run .prompt files.
# Optional kwargs passed by promptmap when declared in the signature:
#   files  - list of open file objects attached to the prompt (via --promptdata key=@path)
#   config - dict of {{ config(...) }} options from the .prompt file (e.g. {"output_format": "json"})
def llm_call(prompt: str, files: list | None = None, config: dict | None = None) -> promptmap.LLMResult:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    content: list = [{"type": "input_text", "text": prompt}]
    for f in (files or []):
        uploaded = client.files.create(file=f, purpose="user_data")
        content.append({"type": "input_file", "file_id": uploaded.id})
    response = client.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        input=[{"role": "user", "content": content}],
    )
    # promptmap.LLMResult reports tokens spent to `promptmap docs`; returning the bare text also works.
    # total_tokens covers input, output and reasoning.
    usage = response.usage
    return promptmap.LLMResult(response.output_text, spent_tokens=usage.total_tokens if usage else None)
""",
    "anthropic": """\
import os
import anthropic

import promptmap

# Automatically picked up by `promptmap` to run .prompt files.
# Optional kwargs passed by promptmap when declared in the signature:
#   files  - list of open file objects attached to the prompt (via --promptdata key=@path)
#   config - dict of {{ config(...) }} options from the .prompt file (e.g. {"output_format": "json"})
def llm_call(prompt: str, files: list | None = None, config: dict | None = None) -> promptmap.LLMResult:
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
    # promptmap.LLMResult reports tokens spent to `promptmap docs`; returning the bare text also works.
    # Anthropic reports no total; output_tokens already includes thinking.
    usage = message.usage
    return promptmap.LLMResult(message.content[0].text, spent_tokens=usage.input_tokens + usage.output_tokens)
""",
}

# ---------------------------------------------------------------------------
# OpenAI-compatible providers: same client code, different endpoint and model.
# `no_thinking` goes in extra_body so a yes/no answer is not eaten by reasoning;
# `top_logprobs` is 0 where the API returns no logprobs.
# ---------------------------------------------------------------------------

OPENAI_COMPATIBLE: dict[str, dict[str, str]] = {
    "deepseek": {
        "name": "DeepSeek", "key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com",
        "model_env": "DEEPSEEK_MODEL", "model": "deepseek-v4-flash",
        "no_thinking": '{"thinking": {"type": "disabled"}}', "top_logprobs": "20",
    },
    "qwen": {
        "name": "Qwen (Alibaba DashScope)", "key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model_env": "QWEN_MODEL", "model": "qwen-plus",
        "no_thinking": '{"enable_thinking": False}', "top_logprobs": "5",
    },
    "kimi": {
        "name": "Kimi (Moonshot)", "key_env": "MOONSHOT_API_KEY", "base_url": "https://api.moonshot.ai/v1",
        "model_env": "KIMI_MODEL", "model": "kimi-k2.6",
        "no_thinking": '{"thinking": {"type": "disabled"}}', "top_logprobs": "0",
    },
    "xiaomi": {
        "name": "Xiaomi MiMo", "key_env": "MIMO_API_KEY", "base_url": "https://api.xiaomimimo.com/v1",
        "model_env": "MIMO_MODEL", "model": "mimo-v2.6-flash",
        "no_thinking": '{"thinking": {"type": "disabled"}}', "top_logprobs": "0",
    },
}


def _fill(template: str, values: dict[str, str]) -> str:
    """Replace <<key>> placeholders (templates hold literal {} and {{ }})."""
    for key, value in values.items():
        template = template.replace(f"<<{key}>>", value)
    return template


_OPENAI_COMPATIBLE_CLIENT = """\
import os
from openai import OpenAI

import promptmap

# Automatically picked up by `promptmap` to run .prompt files.
# <<name>> speaks the OpenAI Chat Completions API.
# Optional kwarg passed by promptmap when declared in the signature:
#   config - dict of {{ config(...) }} options from the .prompt file (e.g. {"output_format": "json"})
MODEL = os.environ.get("<<model_env>>", "<<model>>")


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ["<<key_env>>"], base_url="<<base_url>>")


def llm_call(prompt: str, config: dict | None = None) -> promptmap.LLMResult:
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    # promptmap.LLMResult reports tokens spent to `promptmap docs`; returning the bare text also works.
    # total_tokens covers input, output and reasoning.
    usage = response.usage
    return promptmap.LLMResult(
        response.choices[0].message.content,
        spent_tokens=usage.total_tokens if usage else None,
    )
"""

for _provider, _values in OPENAI_COMPATIBLE.items():
    CLIENT_PY[_provider] = _fill(_OPENAI_COMPATIBLE_CLIENT, _values)


# ---------------------------------------------------------------------------
# Test feedback classifier — appended to client.py by `promptmap init --classifier`.
# Jev-like: ask for ONE token and read P(yes) from its logprobs.  Where the
# provider returns no logprobs the answer is a hard 1.0 / 0.0.
# ---------------------------------------------------------------------------

_CLASSIFIER_HEADER = """\


# --- Test feedback classifier (jev-like) -------------------------------------
# Judges {{ config(judge="classifier") }} tests: question above a `---` line,
# text to judge below it.  Returns P(yes) in [0, 1].
import math


def _p_yes(candidates: list[tuple[str, float]]) -> float:
    \"\"\"P(yes) from (token, logprob) pairs; a plain text answer is (text, 0.0).\"\"\"
    p = {"yes": 0.0, "no": 0.0}
    for token, logprob in candidates:
        word = (token.split() or [""])[0].strip(".,!").lower()
        if word in p:
            p[word] += math.exp(logprob)
    if p["yes"] + p["no"] == 0:
        raise RuntimeError(f"Classifier answered neither yes nor no: {[t for t, _ in candidates]}")
    return p["yes"] / (p["yes"] + p["no"])


def _yes_no_prompt(state: str, question: str) -> str:
    return f"{state}\\n\\n{question}\\nAnswer with only yes or no."

"""

_CLASSIFIER_FOOTER = """

classify_call = classifier_for_test_feedback
"""

_CLASSIFIER_BODY: dict[str, str] = {
    "gemini": """\

def classifier_for_test_feedback(state: str, question: str) -> float:
    \"\"\"Gemini 3 returns no logprobs, so this is a hard 1.0 / 0.0 answer.\"\"\"
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    text = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        contents=_yes_no_prompt(state, question),
    ).text
    return _p_yes([(text or "", 0.0)])
""",
    "openai": """\

def classifier_for_test_feedback(state: str, question: str) -> float:
    \"\"\"One token; P(yes) from its top logprobs (needs a non-reasoning model).\"\"\"
    response = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": _yes_no_prompt(state, question)}],
        max_completion_tokens=1,
        logprobs=True,
        top_logprobs=20,
    )
    top = response.choices[0].logprobs.content[0].top_logprobs
    return _p_yes([(t.token, t.logprob) for t in top])
""",
    "anthropic": """\

def classifier_for_test_feedback(state: str, question: str) -> float:
    \"\"\"Claude returns no logprobs, so this is a hard 1.0 / 0.0 answer.\"\"\"
    message = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]).messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=5,
        messages=[{"role": "user", "content": _yes_no_prompt(state, question)}],
    )
    return _p_yes([(message.content[0].text, 0.0)])
""",
}

_OPENAI_COMPATIBLE_LOGPROBS = """\

def classifier_for_test_feedback(state: str, question: str) -> float:
    \"\"\"One token; P(yes) from its top logprobs.\"\"\"
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": _yes_no_prompt(state, question)}],
        max_tokens=1,
        logprobs=True,
        top_logprobs=<<top_logprobs>>,
        extra_body=<<no_thinking>>,
    )
    top = response.choices[0].logprobs.content[0].top_logprobs
    return _p_yes([(t.token, t.logprob) for t in top])
"""

_OPENAI_COMPATIBLE_TEXT = """\

def classifier_for_test_feedback(state: str, question: str) -> float:
    \"\"\"<<name>> returns no logprobs, so this is a hard 1.0 / 0.0 answer.\"\"\"
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": _yes_no_prompt(state, question)}],
        max_tokens=5,
        extra_body=<<no_thinking>>,
    )
    return _p_yes([(response.choices[0].message.content or "", 0.0)])
"""

for _provider, _values in OPENAI_COMPATIBLE.items():
    _body = _OPENAI_COMPATIBLE_LOGPROBS if _values["top_logprobs"] != "0" else _OPENAI_COMPATIBLE_TEXT
    _CLASSIFIER_BODY[_provider] = _fill(_body, _values)

CLASSIFIER_PY: dict[str, str] = {
    provider: _CLASSIFIER_HEADER + body + _CLASSIFIER_FOOTER
    for provider, body in _CLASSIFIER_BODY.items()
}

PROVIDERS = tuple(CLIENT_PY)


# ---------------------------------------------------------------------------
# CLI command — registered onto the main Click group from cli/__init__.py
# ---------------------------------------------------------------------------

def register_command(main) -> None:
    """Attach the `promptmap init` command to *main* Click group."""
    import click
    from rich.console import Console
    from promptmap.cli.vscode import is_running_in_vscode, setup_vscode_associations

    console = Console()

    @main.command("init")
    @click.argument("project_name", required=False)
    @click.option("--force", is_flag=True, default=False, help="Overwrite existing files.")
    @click.option(
        "--secrets", "with_secrets", is_flag=True, default=False,
        help="Also write secrets.yml + secrets-template.yml and gitignore secrets.yml.",
    )
    @click.option("--data", "with_data", is_flag=True, default=False, help="Also write promptdata.yml.")
    @click.option("--files", "with_files", is_flag=True, default=False, help="Also write promptfiles.yml.")
    @click.option(
        "--models-dir",
        default="models",
        show_default=True,
        help="Models whose keys --secrets/--data/--files write (relative to the project).",
    )
    @click.option(
        "--provider",
        type=click.Choice(PROVIDERS, case_sensitive=False),
        default="gemini",
        show_default=True,
        help="LLM provider to use in the generated client.py.",
    )
    @click.option(
        "--classifier", is_flag=True, default=False,
        help='Add a yes/no classify_call to client.py, for tests judged with config(judge="classifier") or --judge classifier.',
    )
    def init(
        project_name: str | None, force: bool, with_secrets: bool, with_data: bool,
        with_files: bool, models_dir: str, provider: str, classifier: bool,
    ) -> None:
        """Scaffold a starter promptmap project inside PROJECT_NAME/.

        --secrets, --data and --files also write secrets.yml (+ secrets-template.yml),
        promptdata.yml and promptfiles.yml, listing the keys the project's
        models use.  Without PROJECT_NAME they are written into the current
        directory instead of scaffolding a project: `promptmap init --secrets`.
        Existing values are kept; missing keys are appended.

        \b
        Examples:
          promptmap init my_project --provider anthropic
          promptmap init my_project --secrets --data
          promptmap init --secrets          (in an existing project)
        """
        kinds = [k for k, on in (("secrets", with_secrets), ("data", with_data), ("files", with_files)) if on]
        if project_name is None and kinds:
            for kind in kinds:
                init_values_file(console, kind, Path("."), Path(models_dir), force)
            return
        project_name = project_name or "generate_articles_example"
        provider = provider.lower()
        files = dict(INIT_FILES)
        files["client.py"] = CLIENT_PY[provider] + (CLASSIFIER_PY[provider] if classifier else "")
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

        for kind in kinds:
            init_values_file(console, kind, root, root / models_dir, force)

        if created:
            console.print(f"\nRun [bold cyan]cd {project_name} && promptmap run[/bold cyan], or [bold cyan]promptmap run --promptdata topic='your topic'[/bold cyan]")

        if is_running_in_vscode():
            setup_vscode_associations()
            console.print("  [dim]VS Code detected — configured .vscode/settings.json for .prompt files[/dim]")


# ---------------------------------------------------------------------------
# promptmap init --secrets | --data | --files — project-root values files
# ---------------------------------------------------------------------------

_VALUES_HEADERS = {
    "secrets": (
        "Secrets for promptmap run / test / serve — read as call.secrets[\"NAME\"].\n"
        "Never rendered, cached or stored.  Keep this file out of git;\n"
        "commit secrets-template.yml instead.  --secret NAME=VALUE overrides."
    ),
    "template": (
        "Template for secrets.yml (which is gitignored): copy it and fill in values.\n"
        "  cp secrets-template.yml secrets.yml"
    ),
    "data": (
        "Default promptdata() values for promptmap run.\n"
        "--promptdata KEY=VALUE overrides any key here."
    ),
    "files": (
        "Default promptfiles for promptmap run: NAME: path (relative to the project root).\n"
        "--promptfile NAME=PATH overrides any key here."
    ),
}


def _values_keys(kind: str, models_dir: Path) -> list[str]:
    """Keys the project's models use for *kind*; empty if there are no models."""
    from promptmap.executor.graph import get_dag_promptdata, get_dag_promptfiles, load_models
    from promptmap.llm import try_load_client_module
    from promptmap.model_types import get_model_kind

    if not models_dir.is_dir():
        return []
    try_load_client_module(str(models_dir))  # registers project-local kinds and their secrets
    models = load_models(str(models_dir))
    if kind == "data":
        return get_dag_promptdata(models)
    if kind == "files":
        return get_dag_promptfiles(models)
    keys: dict[str, None] = {}
    for spec in models.values():
        model_kind = get_model_kind(spec.model_type)
        for name in sorted(model_kind.required_secrets if model_kind is not None else ()):
            keys[name] = None
    return list(keys)


def _write_values_file(path: Path, keys: list[str], header: str, force: bool) -> tuple[str, list[str]]:
    """Write *keys* to *path*; keep an existing file's values and append missing keys.

    Returns (action, keys added).
    """
    import yaml

    from promptmap.project_values import render_values_file

    if not path.exists() or force:
        action = "overwrote" if path.exists() else "created"
        path.write_text(render_values_file(keys, header), encoding="utf-8")
        return action, keys
    existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    present = {str(k) for k in existing} if isinstance(existing, dict) else set()
    missing = [k for k in keys if k not in present]
    if missing:
        text = path.read_text(encoding="utf-8")
        body = render_values_file(missing, "")
        path.write_text(text + ("" if text.endswith("\n") or not text else "\n") + body, encoding="utf-8")
        return "updated", missing
    return "unchanged", []


def _ensure_gitignored(root: Path, name: str) -> bool:
    """Add *name* to *root*/.gitignore unless already listed. Returns True if added."""
    path = root / ".gitignore"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if name in {line.strip().lstrip("/") for line in text.splitlines()}:
        return False
    sep = "" if not text or text.endswith("\n") else "\n"
    path.write_text(f"{text}{sep}\n# promptmap secrets — never commit\n{name}\n", encoding="utf-8")
    return True


def init_values_file(console, kind: str, root: Path, models_dir: Path, force: bool) -> None:
    """Write *root*'s values file for *kind* (`promptmap init --secrets|--data|--files`)."""
    import sys

    from rich.markup import escape

    from promptmap.project_values import (
        PROMPTDATA_FILE,
        PROMPTFILES_FILE,
        SECRETS_FILE,
        SECRETS_TEMPLATE_FILE,
    )

    try:
        keys = _values_keys(kind, models_dir)
    except Exception as exc:  # noqa: BLE001 — a broken model shouldn't hide the cause
        console.print(f"[red]Error:[/red] could not read models in {models_dir}: {escape(str(exc))}")
        sys.exit(1)

    if kind == "secrets":
        targets = [(SECRETS_TEMPLATE_FILE, "template"), (SECRETS_FILE, "secrets")]
    else:
        targets = [(PROMPTDATA_FILE if kind == "data" else PROMPTFILES_FILE, kind)]

    for filename, header in targets:
        path = root / filename
        action, added = _write_values_file(path, keys, _VALUES_HEADERS[header], force)
        detail = f" ({', '.join(added)})" if added else ""
        style = "dim" if action == "unchanged" else "green"
        console.print(f"  [{style}]{action}[/{style}]  {escape(str(path))}{escape(detail)}")

    if kind == "secrets" and _ensure_gitignored(root, SECRETS_FILE):
        console.print(f"  [green]updated[/green]  {escape(str(root / '.gitignore'))} ({SECRETS_FILE})")
    if not keys:
        console.print(f"  [dim]No {kind} keys found in {models_dir}/ — add name: value lines by hand.[/dim]")
