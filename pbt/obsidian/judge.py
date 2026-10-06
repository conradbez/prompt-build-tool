"""
Decide whether an Obsidian note is a *prompt* (sent to the LLM) or *data*
(information the user wrote down, used as-is).

A data note becomes a ``model_type="template"`` model: its Jinja and links
still render, but no LLM is called and the text itself is the output.

A note marks itself with ``pbt: data`` / ``pbt: prompt`` in its frontmatter (or
a ``pbt:`` config that sets ``model_type``).  Unmarked notes are prompts unless
a judge is used, which asks client.py's ``llm_call`` or ``classify_call``.
Verdicts are cached on the note's text, so an unchanged note is never re-judged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

PROMPT = "prompt"
DATA = "data"
KINDS = (PROMPT, DATA)

#: ``(note_title, note_text) -> "prompt" | "data"``
NoteJudge = Callable[[str, str], str]

DEFAULT_CACHE_PATH = Path(".pbt") / "obsidian_judgements.json"

CLASSIFIER_QUESTION = (
    "Is this note an instruction or task for an AI assistant to carry out, "
    "rather than information, notes or data?"
)

LLM_JUDGE_PROMPT = """\
You are sorting the notes of a prompt pipeline. Each note is either:

- "prompt": an instruction, request or task for an AI model to carry out
  (e.g. "Write a summary of...", "List five ideas for...").
- "data": information the user wrote down to be used as-is - facts, notes,
  reference material, lists, context, a style guide.

Note title: {title}

Note text:
<<<
{text}
>>>

Respond only with valid JSON: {{"kind": "prompt"}} or {{"kind": "data"}}."""


def parse_llm_verdict(output: str) -> str:
    """Return ``"prompt"`` or ``"data"`` from an LLM judge reply; prompt if unclear."""
    text = output.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            kind = str(json.loads(text[start:end + 1]).get("kind", "")).strip().lower()
            if kind in KINDS:
                return kind
        except (ValueError, AttributeError):
            pass
    lowered = text.lower()
    return DATA if DATA in lowered and PROMPT not in lowered else PROMPT


def llm_judge(llm_call: Callable) -> NoteJudge:
    """Judge notes by asking *llm_call* (client.py's, same contract as models)."""
    from pbt.tokens import split_usage

    def judge(title: str, text: str) -> str:
        output = split_usage(llm_call(LLM_JUDGE_PROMPT.format(title=title, text=text)))[0]
        return parse_llm_verdict(str(output))

    return judge


def classifier_judge(classify_call: Callable[[str, str], float], threshold: float = 0.5) -> NoteJudge:
    """Judge notes with ``classify_call(state, question) -> P(yes)``: yes means prompt."""

    def judge(title: str, text: str) -> str:
        score = float(classify_call(f"# {title}\n\n{text}", CLASSIFIER_QUESTION))
        return PROMPT if score >= threshold else DATA

    return judge


def cached(judge: NoteJudge, name: str, cache_path: str | Path = DEFAULT_CACHE_PATH) -> NoteJudge:
    """Wrap *judge* so each verdict is stored in *cache_path*, keyed on judge + note text."""
    path = Path(cache_path)
    try:
        cache: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}

    def judge_cached(title: str, text: str) -> str:
        key = hashlib.sha256("\x00".join((name, title, text)).encode()).hexdigest()
        if cache.get(key) not in KINDS:
            cache[key] = judge(title, text)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
        return cache[key]

    return judge_cached
