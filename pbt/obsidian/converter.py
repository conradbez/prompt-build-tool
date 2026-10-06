"""
Obsidian vault → pbt models.

A separate layer on top of pbt: it reads a folder of Obsidian notes and writes
ordinary ``.prompt`` files, which the regular pbt commands then load.  Nothing
in the executor knows about Obsidian.

Conversion rules
----------------

* Every ``*.md`` note under the vault is a model, named after the note's file
  name: lower-cased, with each run of non-word characters turned into ``_``
  (``Market Research.md`` → ``market_research``).  Hidden folders such as
  ``.obsidian/`` and ``.trash/`` are skipped.
* ``[[Note]]`` and ``![[Note]]`` become ``{{ ref('note') }}``.  Aliases
  (``[[Note|shown]]``) and heading/block anchors (``[[Note#Heading]]``) are
  dropped; ``[[folder/Note]]`` and frontmatter ``aliases`` resolve like
  Obsidian resolves them.  A link to a note that is not in the vault keeps its
  display text and is reported back as unresolved.
* A ``pbt:`` mapping in the frontmatter becomes the model's ``config()``;
  ``[[Note]]`` inside its string values becomes the note's model name, so
  ``each: "[[Ideas]].items[*]"`` works.  ``pbt: false`` leaves a note out.
* A note is a *prompt* (sent to the LLM) or *data* (used as-is, as a
  ``model_type="template"`` model).  ``pbt: data`` / ``pbt: prompt`` says which;
  otherwise a judge decides (see :mod:`pbt.obsidian.judge`), or it is a prompt.
* Obsidian comments (``%% … %%``) are stripped.  Fenced code blocks are copied
  verbatim.  Everything else, Jinja included, passes through unchanged.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pbt.executor.graph import _prompt_name
from pbt.obsidian.judge import DATA, KINDS, PROMPT, NoteJudge

#: Written into a generated models directory; its presence is what allows a
#: rebuild to delete the directory's previous contents.
MARKER_FILENAME = ".pbt-obsidian"

_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)^---[ \t]*(?:\r?\n|\Z)", re.S | re.M)
_FENCE = re.compile(r"(^[ \t]*(```|~~~).*?^[ \t]*\2[^\n]*$)", re.S | re.M)
_COMMENT = re.compile(r"%%.*?%%", re.S)
#: [[target#anchor|alias]], optionally prefixed with ! for an embed.
_LINK = re.compile(r"(!?)\[\[([^\]|#^]*)([#^][^\]|]*)?(?:\|([^\]]*))?\]\]")


class ObsidianError(Exception):
    """The vault cannot be converted (name collision, ambiguous link, ...)."""


@dataclass
class ConvertedNote:
    """One note, converted."""

    name: str                #: pbt model name
    note_path: Path          #: note path, relative to the vault
    prompt_path: Path        #: .prompt path, relative to the models directory
    source: str              #: the .prompt file's contents
    depends_on: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    kind: str = PROMPT       #: "prompt" or "data"
    kind_source: str = "default"  #: "frontmatter", "judge" or "default"


@dataclass
class _Note:
    path: Path
    slug: str
    name: str
    frontmatter: dict
    body: str


def slugify(stem: str) -> str:
    """Return the file-safe, ``ref()``-safe form of a note's file name."""
    slug = re.sub(r"\W+", "_", stem.lower()).strip("_")
    if not slug:
        raise ObsidianError(f"Note name {stem!r} has no letters or digits to name a model after.")
    return slug


def _split_frontmatter(text: str, path: Path) -> tuple[dict, str]:
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ObsidianError(f"{path}: invalid frontmatter: {exc}") from exc
    return (data if isinstance(data, dict) else {}), text[match.end():]


def _read_notes(vault: Path) -> list[_Note]:
    notes: list[_Note] = []
    for path in sorted(vault.rglob("*.md")):
        rel = path.relative_to(vault)
        if any(part.startswith(".") for part in rel.parts):
            continue
        frontmatter, body = _split_frontmatter(path.read_text(encoding="utf-8"), rel)
        if frontmatter.get("pbt") is False:
            continue
        slug = slugify(path.stem)
        notes.append(_Note(rel, slug, _prompt_name(Path(f"{slug}.prompt")), frontmatter, body))
    return notes


class _Resolver:
    """Maps link targets to model names the way Obsidian maps them to notes."""

    def __init__(self, notes: list[_Note]) -> None:
        self._by_path: dict[str, str] = {}
        self._by_title: dict[str, list[str]] = {}
        for note in notes:
            self._by_path[note.path.with_suffix("").as_posix().lower()] = note.name
            titles = [note.path.stem, *_aliases(note.frontmatter)]
            for title in titles:
                names = self._by_title.setdefault(title.lower(), [])
                if note.name not in names:
                    names.append(note.name)

    def resolve(self, target: str) -> str | None:
        key = target.strip().removesuffix(".md").lower()
        if key in self._by_path:
            return self._by_path[key]
        names = self._by_title.get(key.rsplit("/", 1)[-1], [])
        if len(names) > 1:
            raise ObsidianError(
                f"[[{target}]] matches several notes ({', '.join(names)}); "
                "link with the folder path, e.g. [[folder/Note]]."
            )
        return names[0] if names else None


def _aliases(frontmatter: dict) -> list[str]:
    aliases = frontmatter.get("aliases") or frontmatter.get("alias") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    return [str(a) for a in aliases]


def _is_attachment(target: str) -> bool:
    """True for an embedded image/PDF/etc. rather than a note."""
    suffix = Path(target.strip()).suffix.lower()
    return bool(suffix) and suffix != ".md"


def _convert_links(text: str, resolver: _Resolver, as_ref: bool, deps: list[str], unresolved: list[str]) -> str:
    def replace(match: re.Match) -> str:
        bang, target, _anchor, alias = match.groups()
        if not target.strip() or _is_attachment(target):
            return match.group(0)
        name = resolver.resolve(target)
        if name is None:
            if target not in unresolved:
                unresolved.append(target)
            return alias or target
        if name not in deps:
            deps.append(name)
        return f"{{{{ ref('{name}') }}}}" if as_ref else name

    return _LINK.sub(replace, text)


def _convert_body(body: str, resolver: _Resolver, deps: list[str], unresolved: list[str]) -> str:
    parts = _FENCE.split(body)
    out: list[str] = []
    # re.split with two groups yields [text, fence, fence-marker, text, ...].
    i = 0
    while i < len(parts):
        text = _COMMENT.sub("", parts[i])
        out.append(_convert_links(text, resolver, True, deps, unresolved))
        if i + 1 < len(parts):
            out.append(parts[i + 1])
        i += 3
    return "".join(out).strip("\n") + "\n"


def _jinja_literal(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_jinja_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(str(k))}: {_jinja_literal(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))


def _config_call(config: dict, resolver: _Resolver, deps: list[str], unresolved: list[str], path: Path) -> str:
    def links_to_names(value: object) -> object:
        if isinstance(value, str):
            return _convert_links(value, resolver, False, deps, unresolved)
        if isinstance(value, list):
            return [links_to_names(v) for v in value]
        return value

    args = []
    for key, value in config.items():
        if not str(key).isidentifier():
            raise ObsidianError(f"{path}: pbt config key {key!r} is not a valid name.")
        args.append(f"{key}={_jinja_literal(links_to_names(value))}")
    return "{{ config(" + ", ".join(args) + ") -}}\n" if args else ""


def convert_vault(vault: str | Path, judge: NoteJudge | None = None) -> dict[str, ConvertedNote]:
    """Convert every note under *vault*; return ``{model_name: ConvertedNote}``.

    *judge* decides prompt vs data for notes that do not say; without one they
    are prompts.
    """
    vault = Path(vault)
    if not vault.is_dir():
        raise ObsidianError(f"Vault folder '{vault}' not found.")

    notes = _read_notes(vault)
    if not notes:
        raise ObsidianError(f"No *.md notes found in '{vault}'.")

    seen: dict[str, Path] = {}
    for note in notes:
        if note.name in seen:
            raise ObsidianError(
                f"Notes '{seen[note.name]}' and '{note.path}' both become model "
                f"'{note.name}'. Rename one, or exclude it with `pbt: false`."
            )
        seen[note.name] = note.path

    resolver = _Resolver(notes)
    converted: dict[str, ConvertedNote] = {}
    for note in notes:
        deps: list[str] = []
        unresolved: list[str] = []
        config, kind, kind_source = _note_kind(note, judge)
        if kind == DATA:
            config = {**config, "model_type": "template"}
        header = f"{{# Generated by `pbt obsidian` from {note.path.as_posix()} - edit the note, not this file. -#}}\n"
        config_line = _config_call(config, resolver, deps, unresolved, note.path)
        body = _convert_body(note.body, resolver, deps, unresolved)
        converted[note.name] = ConvertedNote(
            name=note.name,
            note_path=note.path,
            prompt_path=note.path.parent / f"{note.slug}.prompt",
            source=header + config_line + body,
            depends_on=deps,
            unresolved=unresolved,
            kind=kind,
            kind_source=kind_source,
        )
    return converted


def _note_kind(note: _Note, judge: NoteJudge | None) -> tuple[dict, str, str]:
    """Return the note's config, its kind, and what decided the kind."""
    marker = note.frontmatter.get("pbt")
    if marker is None:
        marker = {}
    if isinstance(marker, str) and marker.strip().lower() in KINDS:
        return {}, marker.strip().lower(), "frontmatter"
    if not isinstance(marker, dict):
        raise ObsidianError(
            f"{note.path}: frontmatter `pbt:` must be `data`, `prompt`, false, "
            "or a mapping of config() keys."
        )
    if "model_type" in marker:
        return marker, DATA if marker["model_type"] == "template" else PROMPT, "frontmatter"
    if judge is None:
        return marker, PROMPT, "default"
    kind = judge(note.path.stem, _COMMENT.sub("", note.body).strip())
    if kind not in KINDS:
        raise ObsidianError(f"{note.path}: judge returned {kind!r}; expected 'prompt' or 'data'.")
    return marker, kind, "judge"


def load_vault(vault: str | Path, judge: NoteJudge | None = None) -> dict[str, str]:
    """Return ``{model_name: prompt_source}`` for ``pbt.run(models_from_dict=...)``."""
    return {name: note.source for name, note in convert_vault(vault, judge).items()}


def write_models(converted: dict[str, ConvertedNote], out_dir: str | Path) -> list[Path]:
    """Write *converted* into *out_dir*, replacing a previous build there.

    Refuses to touch a non-empty directory this module did not create, so a
    hand-written ``models/`` is never overwritten.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        if (out_dir / MARKER_FILENAME).is_file():
            shutil.rmtree(out_dir)
        elif any(out_dir.iterdir()):
            raise ObsidianError(
                f"'{out_dir}' already exists and was not generated by `pbt obsidian`; "
                "choose another --out directory."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MARKER_FILENAME).write_text(
        "Generated by `pbt obsidian build`; rebuilt from the vault on every run.\n",
        encoding="utf-8",
    )
    written = []
    for note in converted.values():
        path = out_dir / note.prompt_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(note.source, encoding="utf-8")
        written.append(path)
    return written
