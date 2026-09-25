"""
promptparams — named test cases for parameterised ``pbt test`` runs.

Each case is one set of inputs: ``pbt test`` runs the models once per case and
reports every test prompt against that case's outputs.

Where cases live
----------------
``promptparams.yml`` and every ``*.yml`` / ``*.yaml`` file under
``promptparams/`` are loaded and combined (``--promptparams PATH`` replaces
those defaults; it is repeatable and accepts files or directories).  Files are
read in sorted path order, so the case order is stable.

Format
------
::

    # promptparams/base.yml
    baselines:
      default:                       # every case starts from this ...
        promptdata:
          tone: formal
          audience: engineers
        promptfiles:
          document: reports/annual.pdf
      casual:
        extends: default             # baselines can build on each other
        promptdata:
          tone: casual

    cases:
      - name: Formal report for engineers          # ... unless it says otherwise
      - name: Casual tone for developers
        extends: casual
        promptdata:
          audience: developers
      - name: No document attached
        promptfiles:
          document: null             # null removes an inherited key
      - name: Stands alone
        extends: []                  # opt out of the default baseline
        promptdata:
          tone: terse

Rules
-----
* ``name`` is how the case is reported (``tone_test[Casual tone for
  developers]``).  It must be unique across every file.  A case without one is
  named ``<file stem>_<n>``.
* ``extends`` is a baseline name or a list of them, applied left to right with
  the case's own values on top.  A case with no ``extends`` key inherits the
  ``default`` baseline when one exists.
* Baselines are shared by every loaded file, so one file can hold the
  baselines and others only the cases.  Defining a baseline twice is an error.
* ``promptdata`` values may be any YAML value (multi-line strings included).
* ``promptfiles`` values are a path or a list of paths, relative to the YAML
  file that declares them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PATHS = ("promptparams.yml", "promptparams")
DEFAULT_BASELINE = "default"
EXAMPLE_PATH = "promptparams.yml.example"

_FILE_KEYS = {"baselines", "cases"}
_BASELINE_KEYS = {"extends", "promptdata", "promptfiles"}
_CASE_KEYS = _BASELINE_KEYS | {"name"}
_YAML_SUFFIXES = (".yml", ".yaml")


class PromptParamsError(ValueError):
    """A promptparams file is malformed or its cases cannot be resolved."""


@dataclass
class TestCase:
    """One named set of inputs, with its baselines already applied."""

    __test__ = False  # not a pytest class, despite the name

    name: str
    promptdata: dict[str, Any] = field(default_factory=dict)
    promptfiles: dict[str, str | list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_promptparams_files(paths: "list[str | Path] | tuple[str | Path, ...] | None" = None) -> list[Path]:
    """Return the YAML files *paths* name, expanding directories.

    With no *paths*, the defaults (``promptparams.yml`` and ``promptparams/``)
    are used, and ones that do not exist are skipped.  A path given explicitly
    must exist.
    """
    explicit = bool(paths)
    found: list[Path] = []
    for raw in paths or DEFAULT_PATHS:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(
                f for f in p.rglob("*") if f.is_file() and f.suffix in _YAML_SUFFIXES
            ))
        elif p.is_file():
            found.append(p)
        elif explicit:
            raise PromptParamsError(f"promptparams path not found: {p}")
    # The same file named twice (a file and its directory) loads once.
    unique: dict[Path, Path] = {}
    for f in found:
        unique.setdefault(f.resolve(), f)
    return list(unique.values())


def load_cases(paths: "list[str | Path] | tuple[str | Path, ...] | None" = None) -> list[TestCase]:
    """Load, combine and resolve every case in the promptparams files.

    Returns an empty list when there are no files, so callers can treat that
    the same as "no parameterisation requested".
    """
    baselines, raw_cases = _load(paths)
    resolved: list[TestCase] = []
    seen: dict[str, Path] = {}
    for spec, path, idx in raw_cases:
        name = str(spec.get("name") or f"{path.stem}_{idx}").strip()
        if name in seen:
            raise PromptParamsError(
                f"{path}: case name '{name}' is already used in {seen[name]}."
            )
        seen[name] = path
        resolved.append(build_case(name, spec, baselines, where=f"{path}: case '{name}'"))
    return resolved


def load_baselines(paths: "list[str | Path] | tuple[str | Path, ...] | None" = None) -> dict[str, dict]:
    """Return every baseline in the promptparams files, by name."""
    return _load(paths)[0]


def build_case(name: str, spec: dict, baselines: dict[str, dict], where: str = "") -> TestCase:
    """Apply *spec*'s baselines (``default`` when it names none) and return the case."""
    merged = _inherit(spec, baselines, where or f"case '{name}'", default=True, chain=())
    return TestCase(name=name, promptdata=merged["promptdata"], promptfiles=merged["promptfiles"])


def _load(paths) -> tuple[dict[str, dict], list[tuple[dict, Path, int]]]:
    """Parse every file: all baselines by name, and each raw case with its origin."""
    baselines: dict[str, dict] = {}
    defined_in: dict[str, Path] = {}
    raw_cases: list[tuple[dict, Path, int]] = []

    for path in find_promptparams_files(paths):
        doc = _read_yaml(path)
        for name, spec in _mapping(doc.get("baselines"), path, "baselines").items():
            name = str(name)
            spec = {} if spec is None else spec  # `default:` with nothing under it
            if name in baselines:
                raise PromptParamsError(
                    f"{path}: baseline '{name}' is already defined in {defined_in[name]}."
                )
            _check_keys(spec, _BASELINE_KEYS, path, f"baseline '{name}'")
            baselines[name] = _resolve_paths(spec, path)
            defined_in[name] = path
        cases = doc.get("cases") or []
        if not isinstance(cases, list):
            raise PromptParamsError(f"{path}: 'cases' must be a list.")
        for idx, spec in enumerate(cases, start=1):
            _check_keys(spec, _CASE_KEYS, path, f"case {idx}")
            raw_cases.append((_resolve_paths(spec, path), path, idx))
    return baselines, raw_cases


def _inherit(spec: dict, baselines: dict[str, dict], where: str, *, default: bool, chain: tuple) -> dict:
    """Merge *spec* on top of the baselines it extends."""
    if "extends" in spec:
        parents = spec["extends"] or []
        parents = [parents] if isinstance(parents, str) else list(parents)
    elif default and DEFAULT_BASELINE in baselines and DEFAULT_BASELINE not in chain:
        parents = [DEFAULT_BASELINE]
    else:
        parents = []

    merged: dict[str, dict] = {"promptdata": {}, "promptfiles": {}}
    for parent in parents:
        parent = str(parent)
        if parent not in baselines:
            known = ", ".join(sorted(baselines)) or "none defined"
            raise PromptParamsError(f"{where} extends unknown baseline '{parent}' (known: {known}).")
        if parent in chain:
            raise PromptParamsError(
                f"{where}: baselines extend each other in a loop: {' -> '.join((*chain, parent))}."
            )
        inherited = _inherit(baselines[parent], baselines, where, default=False, chain=(*chain, parent))
        for section in merged:
            merged[section].update(inherited[section])

    for section in merged:
        own = _mapping(spec.get(section), where, section)
        for key, value in own.items():
            if value is None:
                merged[section].pop(str(key), None)
            else:
                merged[section][str(key)] = value
    return merged


def _read_yaml(path: Path) -> dict:
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PromptParamsError(f"{path}: invalid YAML: {exc}") from exc
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise PromptParamsError(f"{path}: expected a mapping with 'baselines' and/or 'cases'.")
    _check_keys(doc, _FILE_KEYS, path, "file")
    return doc


def _mapping(value: Any, where: "Path | str", what: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PromptParamsError(f"{where}: '{what}' must be a mapping.")
    return value


def _check_keys(spec: Any, allowed: set[str], path: "Path | str", what: str) -> None:
    if not isinstance(spec, dict):
        raise PromptParamsError(f"{path}: {what} must be a mapping.")
    unknown = sorted(set(map(str, spec)) - allowed)
    if unknown:
        raise PromptParamsError(
            f"{path}: {what} has unknown key(s) {', '.join(unknown)} "
            f"(allowed: {', '.join(sorted(allowed))})."
        )


def _resolve_paths(spec: dict, path: Path) -> dict:
    """Return *spec* with its promptfiles made relative to the working directory."""
    files = _mapping(spec.get("promptfiles"), path, "promptfiles")
    if not files:
        return spec

    def resolve(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value:
            raise PromptParamsError(
                f"{path}: promptfile '{name}' must be a path or a list of paths."
            )
        target = Path(value).expanduser()
        if not target.is_absolute():
            target = path.parent / target
        return _display_path(target)

    out: dict = {}
    for name, value in files.items():
        if value is None:
            out[name] = None
        elif isinstance(value, list):
            out[name] = [resolve(v, name) for v in value]
        else:
            out[name] = resolve(value, name)
    return {**spec, "promptfiles": out}


def _display_path(target: Path) -> str:
    """*target* relative to the working directory when it is inside it."""
    try:
        return os.path.relpath(target)
    except ValueError:  # another drive on Windows
        return str(target)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def save_case(
    directory: str | Path,
    name: str,
    promptdata: dict[str, Any],
    promptfiles: dict[str, str | list[str]],
    extends: list[str] | None = None,
) -> Path:
    """Write one case to its own file in *directory* and return the path.

    One file per case means saving never rewrites (or loses the comments in)
    an existing file — the loader combines them anyway.  Promptfile paths,
    given relative to the working directory, are rewritten relative to the
    new file.  Refuses to overwrite a file that already exists.
    """
    directory = Path(directory)
    target = directory / f"{_slug(name)}.yml"
    if target.exists():
        raise PromptParamsError(f"{target} already exists; pick another case name.")

    def rel(p: str) -> str:
        return os.path.relpath(Path(p).expanduser().resolve(), directory.resolve())

    case: dict[str, Any] = {"name": name}
    if extends is not None:
        case["extends"] = extends
    if promptdata:
        case["promptdata"] = dict(promptdata)
    if promptfiles:
        case["promptfiles"] = {
            k: [rel(p) for p in v] if isinstance(v, list) else rel(v)
            for k, v in promptfiles.items()
        }
    directory.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump({"cases": [case]}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "case"


def write_example(
    path: str | Path,
    promptdata_keys: list[str],
    promptfiles_keys: list[str],
) -> None:
    """
    Write a ``promptparams.yml.example`` template for the given inputs.

    It shows a ``default`` baseline holding every input and two cases that
    build on it.  Nothing is written when there are no inputs to show.
    """
    if not promptdata_keys and not promptfiles_keys:
        return

    def block(indent: str) -> list[str]:
        lines: list[str] = []
        if promptdata_keys:
            lines.append(f"{indent}promptdata:")
            lines += [f"{indent}  {k}: <{k}>" for k in promptdata_keys]
        if promptfiles_keys:
            lines.append(f"{indent}promptfiles:")
            lines += [f"{indent}  {k}: <path/to/{k}>" for k in promptfiles_keys]
        return lines

    first_key = promptdata_keys[0] if promptdata_keys else None
    override = (
        ["    promptdata:", f"      {first_key}: <another {first_key}>"]
        if first_key
        else ["    promptfiles:", f"      {promptfiles_keys[0]}: <path/to/another>"]
    )
    lines = [
        "# Copy to promptparams.yml, or any *.yml in promptparams/ (they are combined).",
        "baselines:",
        "  default:            # every case starts from this unless it sets `extends`",
        *block("    "),
        "",
        "cases:",
        "  - name: Baseline inputs",
        "  - name: <describe what this case checks>",
        *override,
        "",
    ]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
