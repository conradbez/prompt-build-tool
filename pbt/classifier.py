"""
Classifier judging for ``pbt test``.

A classifier (such as TypeSafe's Jev) answers a yes/no question about some
text with a probability instead of generating a reply.  A classifier test
puts the question above a ``---`` line and the text being judged below it::

    Does this haiku have exactly three lines?
    ---
    {{ ref('haiku') }}

The test passes when P(yes) reaches the threshold (0.5 unless the test sets
``{{ config(threshold=0.8) }}``).

The backend is ``classify_call(state: str, question: str) -> float`` in
client.py.  :func:`systemone_classifier` builds one for any service speaking
the ``/v1/systemone`` API — TypeSafe's hosted Jev, or a local Ollaya server —
using only the standard library.

Which judge a test uses: its own ``config(judge=...)``, else the run's judge
(``pbt test --judge``, else ``test_judge`` in client.py, else ``"llm"``).
LLM-judged tests are unchanged: the whole rendered file, ``---`` included,
goes to ``llm_call``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable

JUDGES = ("llm", "classifier")
DEFAULT_THRESHOLD = 0.5
SEPARATOR = "---"

ClassifyCall = Callable[[str, str], float]


def split_question(rendered: str) -> tuple[str, str]:
    """
    Split a rendered classifier test into ``(question, state)`` at the first
    line that is exactly ``---``.

    Raises ValueError when there is no separator or either side is empty.
    """
    lines = rendered.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == SEPARATOR:
            question = "\n".join(lines[:i]).strip()
            state = "\n".join(lines[i + 1:]).strip()
            if not question:
                raise ValueError("Classifier test has no question above the '---' line.")
            if not state:
                raise ValueError("Classifier test has nothing to judge below the '---' line.")
            return question, state
    raise ValueError(
        "Classifier test needs a '---' line: the question goes above it and "
        "the text to judge below it."
    )


def parse_judge(value: str, where: str) -> str:
    """Validate a judge name, naming *where* it came from on error."""
    if value not in JUDGES:
        raise ValueError(f"{where}: judge must be one of {', '.join(JUDGES)}; got {value!r}.")
    return value


def parse_threshold(value: str | None) -> float:
    """Return the pass threshold from a config() value (a string), default 0.5."""
    if value is None:
        return DEFAULT_THRESHOLD
    try:
        threshold = float(value)
    except ValueError:
        raise ValueError(f"threshold must be a number between 0 and 1; got {value!r}.") from None
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1; got {threshold}.")
    return threshold


def systemone_classifier(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> ClassifyCall:
    """
    Return a ``classify_call`` for a ``/v1/systemone`` endpoint.

    Defaults follow TypeSafe's SDK: ``TYPESAFE_BASE_URL`` (else
    ``https://api.typesafe.ai``), ``TYPESAFE_DEFAULT_MODEL`` (else
    ``jev-latest``) and ``TYPESAFE_API_KEY``.  For a local Ollaya server::

        classify_call = pbt.systemone_classifier(
            model="laya:en", base_url="http://localhost:11435"
        )
    """
    base = (base_url or os.environ.get("TYPESAFE_BASE_URL") or "https://api.typesafe.ai").rstrip("/")
    model = model or os.environ.get("TYPESAFE_DEFAULT_MODEL") or "jev-latest"
    key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY")

    def classify_call(state: str, question: str) -> float:
        body = json.dumps({
            "model": model,
            "state": state,
            "questions": {"pass": {"type": "noul", "instructions": question}},
        }).encode()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(f"{base}/v1/systemone", data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"{base}/v1/systemone returned {exc.code}: {detail}") from None
        try:
            return float(data["answers"]["pass"]["noul"])
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(f"Unexpected /v1/systemone response: {data!r}"[:500]) from None

    return classify_call
