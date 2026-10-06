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
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import unquote

import yaml

from pbt.executor.graph import _prompt_name
from pbt.obsidian.judge import DATA, KINDS, PROMPT, NoteInfo, NoteJudge

#: Written into a generated models directory; its presence is what allows a
#: rebuild to delete the directory's previous contents.
MARKER_FILENAME = ".pbt-obsidian"

_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)^---[ \t]*(?:\r?\n|\Z)", re.S | re.M)
_FENCE = re.compile(r"(^[ \t]*(```|~~~).*?^[ \t]*\2[^\n]*$)", re.S | re.M)
_COMMENT = re.compile(r"%%.*?%%", re.S)
#: [[target#anchor|alias]], optionally prefixed with ! for an embed.
_LINK = re.compile(r"(!?)\[\[([^\]|#^]*)([#^][^\]|]*)?(?:\|([^\]]*))?\]\]")
#: [shown](path/to/note.md "title"), optionally prefixed with ! for an embed.
_MD_LINK = re.compile(r"""(!?)\[([^\]]*)\]\(\s*(<[^>]*>|[^)\s]+)(?:\s+["'][^)]*["'])?\s*\)""")


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
    path: Path      #: relative to the vault, or to the folder holding every note
    abs: Path       #: resolved absolute path
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


def _make_note(path: Path, root: Path) -> _Note | None:
    """Read *path*; None when its frontmatter says ``pbt: false``."""
    rel = path.relative_to(root)
    frontmatter, body = _split_frontmatter(path.read_text(encoding="utf-8"), rel)
    if frontmatter.get("pbt") is False:
        return None
    slug = slugify(path.stem)
    return _Note(rel, path.resolve(), slug, _prompt_name(Path(f"{slug}.prompt")), frontmatter, body)


def _read_vault(vault: Path) -> list[_Note]:
    """Every note under *vault*, hidden folders (.obsidian, .trash) aside."""
    notes: list[_Note] = []
    for path in sorted(vault.rglob("*.md")):
        if any(part.startswith(".") for part in path.relative_to(vault).parts):
            continue
        note = _make_note(path, vault)
        if note is not None:
            notes.append(note)
    return notes


def _md_href(href: str) -> str | None:
    """The note path a markdown link points to, or None for URLs and non-notes."""
    href = unquote(href.strip().strip("<>")).split("#", 1)[0]
    if not href or re.match(r"^[a-z][a-z0-9+.-]*:", href, re.I) or not href.lower().endswith(".md"):
        return None
    return href


def _relative_note(from_path: Path, target: str) -> Path | None:
    """*target* (a link written in the note at *from_path*) resolved from that note's folder."""
    target = target.strip()
    if not target or target.startswith("/"):
        return None
    if not target.lower().endswith(".md"):
        target += ".md"
    return (from_path.parent / target).resolve()


def _link_targets(text: str) -> list[str]:
    """Every note link target in *text*, outside code fences and comments."""
    targets = []
    for i, part in enumerate(_FENCE.split(text)):
        if i % 3:
            continue  # a fence, or its marker
        part = _COMMENT.sub("", part)
        targets += [m.group(2) for m in _LINK.finditer(part) if not _is_attachment(m.group(2))]
        targets += [t for t in (_md_href(m.group(3)) for m in _MD_LINK.finditer(part)) if t]
    return targets


def _read_linked(entry: Path) -> list[_Note]:
    """*entry* and every note reachable from it through links.

    Each link resolves relative to the folder of the note it is written in, so
    no vault is needed: plain markdown files that link to each other work.
    """
    found: dict[Path, tuple[str, dict]] = {}
    queue = [entry.resolve()]
    while queue:
        path = queue.pop(0)
        if path in found:
            continue
        text = path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text, path)
        if frontmatter.get("pbt") is False and path != entry.resolve():
            continue
        found[path] = (body, frontmatter)
        config = frontmatter.get("pbt")
        for target in _link_targets(body + "\n" + (json.dumps(config) if isinstance(config, dict) else "")):
            linked = _relative_note(path, target)
            if linked is not None and linked.is_file() and linked not in found:
                queue.append(linked)

    root = Path(os.path.commonpath([p.parent for p in found]))
    notes = [_make_note(path, root) for path in found]
    return [note for note in notes if note is not None]


class _Resolver:
    """Maps link targets to model names.

    A link is first resolved relative to the folder of the note it is written
    in.  In a vault, it then falls back to Obsidian's rules: a path from the
    vault root, then a note title or alias anywhere in the vault.
    """

    def __init__(self, notes: list[_Note], vault_lookup: bool) -> None:
        self._by_abs = {note.abs: note.name for note in notes}
        self._vault_lookup = vault_lookup
        self._by_path: dict[str, str] = {}
        self._by_title: dict[str, list[str]] = {}
        for note in notes:
            self._by_path[note.path.with_suffix("").as_posix().lower()] = note.name
            titles = [note.path.stem, *_aliases(note.frontmatter)]
            for title in titles:
                names = self._by_title.setdefault(title.lower(), [])
                if note.name not in names:
                    names.append(note.name)

    def for_note(self, note: _Note) -> Callable[[str], str | None]:
        """Return a resolver for links written in *note*."""
        return lambda target: self.resolve(target, note)

    def resolve(self, target: str, from_note: _Note) -> str | None:
        relative = _relative_note(from_note.abs, target)
        if relative in self._by_abs:
            return self._by_abs[relative]
        if not self._vault_lookup:
            return None
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


Resolve = Callable[[str], "str | None"]


def _convert_links(text: str, resolve: Resolve, as_ref: bool, deps: list[str], unresolved: list[str]) -> str:
    def link(target: str, shown: str) -> str:
        name = resolve(target)
        if name is None:
            if target not in unresolved:
                unresolved.append(target)
            return shown
        if name not in deps:
            deps.append(name)
        return f"{{{{ ref('{name}') }}}}" if as_ref else name

    def wikilink(match: re.Match) -> str:
        _bang, target, _anchor, alias = match.groups()
        if not target.strip() or _is_attachment(target):
            return match.group(0)
        return link(target, alias or target)

    def md_link(match: re.Match) -> str:
        _bang, shown, href = match.groups()
        target = _md_href(href)
        return match.group(0) if target is None else link(target, shown or target)

    return _MD_LINK.sub(md_link, _LINK.sub(wikilink, text))


def _convert_body(body: str, resolver: Resolve, deps: list[str], unresolved: list[str]) -> str:
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


def _config_call(config: dict, resolver: Resolve, deps: list[str], unresolved: list[str], path: Path) -> str:
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


def convert(source: str | Path, judge: NoteJudge | None = None) -> dict[str, ConvertedNote]:
    """Convert a vault folder, or one ``.md`` note and the notes it links to.

    Returns ``{model_name: ConvertedNote}``.  *judge* decides prompt vs data
    for notes that do not say; without one they are prompts.
    """
    source = Path(source)
    if source.is_dir():
        return convert_vault(source, judge)
    if source.is_file() and source.suffix.lower() == ".md":
        return _convert_notes(_read_linked(source), judge, vault_lookup=False)
    raise ObsidianError(f"'{source}' is neither a folder nor a .md file.")


def convert_vault(vault: str | Path, judge: NoteJudge | None = None) -> dict[str, ConvertedNote]:
    """Convert every note under *vault*; return ``{model_name: ConvertedNote}``.

    *judge* decides prompt vs data for notes that do not say; without one they
    are prompts.
    """
    vault = Path(vault)
    if not vault.is_dir():
        raise ObsidianError(f"Vault folder '{vault}' not found.")

    notes = _read_vault(vault)
    if not notes:
        raise ObsidianError(f"No *.md notes found in '{vault}'.")
    return _convert_notes(notes, judge, vault_lookup=True)


def _convert_notes(notes: list[_Note], judge: NoteJudge | None, vault_lookup: bool) -> dict[str, ConvertedNote]:
    seen: dict[str, Path] = {}
    for note in notes:
        if note.name in seen:
            raise ObsidianError(
                f"Notes '{seen[note.name]}' and '{note.path}' both become model "
                f"'{note.name}'. Rename one, or exclude it with `pbt: false`."
            )
        seen[note.name] = note.path

    resolver = _Resolver(notes, vault_lookup)
    converted: dict[str, ConvertedNote] = {}
    # First pass: links only, so the judge can see every note's backlinks.
    parsed = {}
    for note in notes:
        deps: list[str] = []
        unresolved: list[str] = []
        config = _note_config(note)
        resolve = resolver.for_note(note)
        config_line = _config_call(config if isinstance(config, dict) else {}, resolve, deps, unresolved, note.path)
        body = _convert_body(note.body, resolve, deps, unresolved)
        parsed[note.name] = (config, config_line, body, deps, unresolved)

    titles = {note.name: note.path.stem for note in notes}
    for note in notes:
        config, config_line, body, deps, unresolved = parsed[note.name]
        info = NoteInfo(
            title=note.path.stem,
            text=_COMMENT.sub("", note.body).strip(),
            folder=note.path.parent.as_posix() if note.path.parent != Path(".") else "",
            links_to=[titles[d] for d in deps],
            linked_from=[titles[n] for n, p in parsed.items() if note.name in p[3]],
        )
        kind, kind_source = _note_kind(note, config, info, judge)
        mapping = config if isinstance(config, dict) else {}
        if kind == DATA and "model_type" not in mapping:
            config_line = _config_call({**mapping, "model_type": "template"}, resolver.for_note(note), [], [], note.path)
        header = f"{{# Generated by `pbt obsidian` from {note.path.as_posix()} - edit the note, not this file. -#}}\n"
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


def _note_config(note: _Note) -> dict | str:
    """Return the frontmatter ``pbt:`` mapping, or ``"data"`` / ``"prompt"``."""
    marker = note.frontmatter.get("pbt")
    if marker is None:
        return {}
    if isinstance(marker, str) and marker.strip().lower() in KINDS:
        return marker.strip().lower()
    if not isinstance(marker, dict):
        raise ObsidianError(
            f"{note.path}: frontmatter `pbt:` must be `data`, `prompt`, false, "
            "or a mapping of config() keys."
        )
    return marker


def _note_kind(note: _Note, config: dict | str, info: NoteInfo, judge: NoteJudge | None) -> tuple[str, str]:
    """Return the note's kind and what decided it."""
    if isinstance(config, str):
        return config, "frontmatter"
    if "model_type" in config:
        return (DATA if config["model_type"] == "template" else PROMPT), "frontmatter"
    if judge is None:
        return PROMPT, "default"
    kind = judge(info)
    if kind not in KINDS:
        raise ObsidianError(f"{note.path}: judge returned {kind!r}; expected 'prompt' or 'data'.")
    return kind, "judge"


def load_vault(source: str | Path, judge: NoteJudge | None = None) -> dict[str, str]:
    """Return ``{model_name: prompt_source}`` for ``pbt.run(models_from_dict=...)``.

    *source* is a vault folder or a single ``.md`` note (see :func:`convert`).
    """
    return {name: note.source for name, note in convert(source, judge).items()}


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
