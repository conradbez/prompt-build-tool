"""
promptparams.csv support — parameterised test runs.

Format
------
Each row in ``promptparams.csv`` provides one set of parameters for a test run.
Tests are *cross-joined* with rows: every test prompt runs once per row.

Column naming convention
------------------------
``promptdata.<key>``
    Injected via ``promptdata("<key>")`` in Jinja2 templates.

``promptfile.<name>``
    Provided as a file reference to ``llm_call`` (if it accepts a ``files``
    parameter).  The cell value may be:

    * a single path – ``report.pdf``
    * a JSON array  – ``["a.pdf", "b.pdf"]``

Example ``promptparams.csv``
----------------------------
::

    promptdata.tone,promptdata.audience,promptfile.document
    formal,engineers,reports/annual.pdf
    casual,developers,"[""report1.pdf"", ""report2.pdf""]"
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROMPTDATA_PREFIX = "promptdata."
PROMPTFILE_PREFIX = "promptfile."


def load_promptparams(path: str | Path = "promptparams.csv") -> list[dict[str, str]]:
    """
    Load ``promptparams.csv`` and return a list of raw row dicts (one per row).

    Returns an empty list when the file does not exist, so callers can treat
    a missing file the same as "no parameterisation requested".
    """
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def parse_promptparams_row(
    row: dict[str, str],
) -> tuple[dict[str, str], dict[str, str | list[str]]]:
    """
    Split a CSV row into ``(promptdata_dict, promptfiles_dict)``.

    Column mapping:

    * ``promptdata.<key>``  →  ``promptdata_dict[key] = value``
    * ``promptfile.<name>`` →  ``promptfiles_dict[name] = path_or_list``

    For ``promptfile.*`` columns, the cell value may be a plain path string
    or a JSON-encoded list of paths.  Empty cells are skipped.
    """
    promptdata: dict[str, str] = {}
    promptfiles: dict[str, str | list[str]] = {}

    for col, value in row.items():
        if not value:
            continue
        if col.startswith(PROMPTDATA_PREFIX):
            key = col[len(PROMPTDATA_PREFIX):]
            promptdata[key] = value
        elif col.startswith(PROMPTFILE_PREFIX):
            name = col[len(PROMPTFILE_PREFIX):]
            stripped = value.strip()
            if stripped.startswith("["):
                try:
                    promptfiles[name] = json.loads(stripped)
                except json.JSONDecodeError:
                    promptfiles[name] = stripped
            else:
                promptfiles[name] = stripped

    return promptdata, promptfiles


def get_promptparams_columns(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    """
    Return ``(promptdata_keys, promptfiles_keys)`` found across all rows.

    Keys are deduplicated while preserving first-seen order.
    """
    pd_seen: dict[str, None] = {}
    pf_seen: dict[str, None] = {}
    for row in rows:
        for col in row:
            if col.startswith(PROMPTDATA_PREFIX):
                pd_seen[col[len(PROMPTDATA_PREFIX):]] = None
            elif col.startswith(PROMPTFILE_PREFIX):
                pf_seen[col[len(PROMPTFILE_PREFIX):]] = None
    return list(pd_seen), list(pf_seen)


def write_example(
    path: str | Path,
    promptdata_keys: list[str],
    promptfiles_keys: list[str],
) -> None:
    """
    Write a ``promptparams.csv.example`` template with the given column headers.

    Creates one placeholder row so users know the expected format.
    Nothing is written when there are no columns to show.
    """
    headers = (
        [f"{PROMPTDATA_PREFIX}{k}" for k in promptdata_keys]
        + [f"{PROMPTFILE_PREFIX}{k}" for k in promptfiles_keys]
    )
    if not headers:
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        example_row = (
            [f"<{k}>" for k in promptdata_keys]
            + [f"<path/to/{k}>" for k in promptfiles_keys]
        )
        writer.writerow(example_row)
