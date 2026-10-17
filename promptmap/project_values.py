"""
Project-root defaults for run inputs: ``secrets.yml``, ``promptdata.yml`` and
``promptfiles.yml``.

Each is a flat YAML mapping of name → value, read from the directory promptmap runs
in.  ``promptmap run`` uses them for any value not passed on the command line —
``--secret``, ``--promptdata`` and ``--promptfile`` always win.  Paths in
``promptfiles.yml`` are relative to the project root.

``secrets.yml`` holds credentials, so it belongs in ``.gitignore``; commit
``secrets-template.yml`` (same keys, empty values) instead.  ``promptmap init
--secrets`` / ``--data`` / ``--files`` writes a file with the keys the
project's models use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

SECRETS_FILE = "secrets.yml"
SECRETS_TEMPLATE_FILE = "secrets-template.yml"
PROMPTDATA_FILE = "promptdata.yml"
PROMPTFILES_FILE = "promptfiles.yml"


class ProjectValuesError(ValueError):
    """A project-root values file is not a flat mapping of names to values."""


def load_values(path: str | Path) -> dict[str, str]:
    """Read a flat ``name: value`` YAML file; a missing file is empty.

    Keys with an empty or null value are left out, so a file of blank keys
    (like a fresh template) supplies nothing.  Errors never echo a value.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        raise ProjectValuesError(f"{path} is not valid YAML{where}") from None
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise ProjectValuesError(f"{path} must be a mapping of name: value")
    values: dict[str, str] = {}
    for key, value in doc.items():
        if isinstance(value, (dict, list)):
            raise ProjectValuesError(f"{path}: {key!r} must be a single value, not a {type(value).__name__}")
        if value is None or value == "":
            continue
        values[str(key)] = str(value)
    return values


def render_values_file(keys: Iterable[str], header: str) -> str:
    """Return YAML text listing *keys* with empty values, under a comment *header*."""
    lines = [f"# {line}" if line else "#" for line in header.splitlines()]
    keys = list(dict.fromkeys(keys))
    if keys:
        lines += [f"{yaml.safe_dump(k).splitlines()[0]}: \"\"" for k in keys]
    else:
        lines.append("# (no keys found — add name: value lines here)")
    return "\n".join(lines) + "\n"
