"""
Decide whether an Obsidian note is a *prompt* (sent to the LLM) or *data*
(information the user wrote down, used as-is).

A data note becomes a ``model_type="template"`` model: its Jinja and links
still render, but no LLM is called and the text itself is the output.

A note marks itself with ``pbt: data`` / ``pbt: prompt`` in its frontmatter (or
a ``pbt:`` config that sets ``model_type``).  Unmarked notes are prompts unless
a judge is used, which asks client.py's ``llm_call`` or ``classify_call``.

The judge is told the situation (:data:`VAULT_CONTEXT`: reference material the
user keeps vs. hypotheses, ideas and questions they want input on), plus any
project description the user supplies, and sees the note's title, folder, the
notes linking to it and the notes it links to.  Verdicts are cached on all of
that, so an unchanged note is never re-judged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROMPT = "prompt"
DATA = "data"
KINDS = (PROMPT, DATA)

DEFAULT_CACHE_PATH = Path(".pbt") / "obsidian_judgements.json"


@dataclass
class NoteInfo:
    """What a judge sees of one note: its text and where it sits in the vault."""

    title: str
    text: str
    folder: str = ""
    links_to: list[str] = field(default_factory=list)     #: titles of notes it links to
    linked_from: list[str] = field(default_factory=list)  #: titles of notes linking to it

    def describe(self) -> str:
        """The note as shown to the judge: title, place in the vault, then text."""
        lines = [f"Note title: {self.title}"]
        if self.folder:
            lines.append(f"Folder: {self.folder}")
        lines.append("Used by (notes that link to this one): " + (", ".join(self.linked_from) or "none"))
        lines.append("Uses (notes this one links to): " + (", ".join(self.links_to) or "none"))
        return "\n".join(lines) + f"\n\nNote text:\n<<<\n{self.text}\n>>>"


#: ``(note) -> "prompt" | "data"``
NoteJudge = Callable[[NoteInfo], str]

#: The situation both judges are told about.
VAULT_CONTEXT = """\
These notes are texts a user gave us from their notes vault, which runs as an
AI pipeline. The user writes two sorts of note:

- Reference material they keep so other notes can use it later: facts,
  sources, background, quotes, definitions, data, a style guide or brief.
  It is passed along as-is and never sent to the AI on its own.
- Notes they want an AI's input on: a hypothesis to test or challenge, an
  idea to develop, an open question, a draft to improve, a task or
  instruction. These are sent to the AI, together with any notes they use.

Signals: a note other notes link to is usually reference material; a note
that links to others and asks, proposes or instructs usually wants input.
A hypothesis stated as fact still wants input if the user is exploring it."""

CLASSIFIER_QUESTION = (
    "Does the user want an AI's input on this note (a hypothesis, idea, question, "
    "draft or task), rather than keeping it as reference material for other notes?"
)

LLM_JUDGE_PROMPT = """\
{context}

Decide which sort this note is:

- "prompt": the user wants an AI's input on it.
- "data": reference material, used as-is by other notes.

{note}

Respond only with valid JSON: {{"kind": "prompt"}} or {{"kind": "data"}}."""


def _context(extra: str | None) -> str:
    if not extra:
        return VAULT_CONTEXT
    return f"{VAULT_CONTEXT}\n\nMore about this project, from the user:\n{extra.strip()}"


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


def llm_judge(llm_call: Callable, context: str | None = None) -> NoteJudge:
    """Judge notes by asking *llm_call* (client.py's, same contract as models).

    *context* is extra project description from the user, added to the
    built-in :data:`VAULT_CONTEXT`.
    """
    from pbt.tokens import split_usage

    def judge(note: NoteInfo) -> str:
        prompt = LLM_JUDGE_PROMPT.format(context=_context(context), note=note.describe())
        return parse_llm_verdict(str(split_usage(llm_call(prompt))[0]))

    return judge


def classifier_judge(
    classify_call: Callable[[str, str], float],
    context: str | None = None,
    threshold: float = 0.5,
) -> NoteJudge:
    """Judge notes with ``classify_call(state, question) -> P(yes)``: yes means prompt.

    The state is the situation (:data:`VAULT_CONTEXT` plus *context*) followed
    by the note; the question is :data:`CLASSIFIER_QUESTION`.
    """

    def judge(note: NoteInfo) -> str:
        state = f"{_context(context)}\n\n{note.describe()}"
        score = float(classify_call(state, CLASSIFIER_QUESTION))
        return PROMPT if score >= threshold else DATA

    return judge


def cached(judge: NoteJudge, name: str, cache_path: str | Path = DEFAULT_CACHE_PATH) -> NoteJudge:
    """Wrap *judge* so each verdict is stored in *cache_path*.

    The key is *name* plus everything the judge sees of the note, so editing
    the note or its links re-judges it.  Put anything else that changes the
    verdict (such as the user's context) into *name*.
    """
    path = Path(cache_path)
    try:
        cache: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}

    def judge_cached(note: NoteInfo) -> str:
        key = hashlib.sha256("\x00".join((name, note.describe())).encode()).hexdigest()
        if cache.get(key) not in KINDS:
            cache[key] = judge(note)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
        return cache[key]

    return judge_cached
