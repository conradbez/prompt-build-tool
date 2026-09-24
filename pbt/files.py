"""
Files as model outputs — :class:`File`, :class:`Dir` and :class:`Output`.

A model may produce files instead of, or alongside, text.  ``llm_call`` (or an
``execute_python`` model, or a custom kind) returns one of these, or any
dict/list with them inside::

    return pbt.File(png_bytes, name="logo.png")
    return pbt.Dir("build/site")
    return pbt.Output("A fox logo, flat style.", files=[pbt.File(png_bytes, name="logo.png")])
    return {"caption": text, "image": pbt.File(png_bytes, name="logo.png")}

The executor stores each file's bytes once in a content-addressed
:class:`BlobStore` (keyed by sha256) and keeps a small manifest in the model's
output.  Downstream, ``{{ ref('logo') }}`` renders a one-line handle naming the
file and its hash, ``{{ config(promptfiles=["logo"]) }}`` attaches the bytes
to ``llm_call(files=[...])``, and Python code gets the objects themselves.

Stored form, and why it is safe
-------------------------------
Storage only holds strings.  An output with files in it is stored as
``$pbt:v1`` + newline + JSON, where each file is an object tagged with the
``"$pbt"`` marker key.  Only :func:`encode_output` writes that form, and it is
the only place markers are created, so a marker can only come from a real
:class:`File` the executor stored:

* Any dict key spelled ``$pbt`` (or ``$$pbt``, …) in model data is escaped by
  one extra ``$`` on the way in and restored on the way out, so JSON a model
  wrote can never be read back as a file reference.
* A plain text output that happens to start with ``$pbt:`` is itself escaped
  with a ``$pbt:str`` prefix, so text can never pose as an encoded output.
* :func:`decode_output` still validates every marker it reads — hash format,
  file name, relative paths, MIME type, the directory's tree hash — and file
  reads verify the bytes against their hash.

The blob store is pluggable: implement ``put``/``get``/``exists`` (see
:class:`BlobStore`) to keep bytes in S3 or anywhere else.
"""

from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
import re
import sqlite3
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import IO, Any, Iterator, Protocol, Union, runtime_checkable

#: The marker key that tags a file object inside a stored output.
MARKER = "$pbt"

#: Prefix of a stored output that carries files.
ENVELOPE_PREFIX = "$pbt:v1\n"

#: Prefix that escapes a plain text output which itself starts with "$pbt:".
_TEXT_ESCAPE_PREFIX = "$pbt:str\n"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MIME = re.compile(r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*")
_MARKER_KEY = re.compile(r"\$+pbt")
_ESCAPED_MARKER_KEY = re.compile(r"\$\$+pbt")

#: Longest directory listing rendered into a prompt by ``str(Dir)``.
_LISTING_LIMIT = 50


class FileOutputError(ValueError):
    """A stored file reference failed validation, or its bytes are missing."""


class BlobMissingError(FileOutputError):
    """A stored output names a blob the blob store no longer has."""


# ---------------------------------------------------------------------------
# Blob stores
# ---------------------------------------------------------------------------

@runtime_checkable
class BlobStore(Protocol):
    """Content-addressed byte storage, keyed by lowercase hex sha256.

    Three methods are the whole interface, so a store backed by S3, GCS or a
    shared filesystem is a small class::

        class S3BlobStore:
            def __init__(self, bucket, prefix="pbt/blobs/"):
                import boto3
                self.s3, self.bucket, self.prefix = boto3.client("s3"), bucket, prefix

            def put(self, sha256, data):
                if not self.exists(sha256):
                    self.s3.put_object(Bucket=self.bucket, Key=self.prefix + sha256, Body=data)

            def get(self, sha256):
                try:
                    obj = self.s3.get_object(Bucket=self.bucket, Key=self.prefix + sha256)
                except self.s3.exceptions.NoSuchKey:
                    raise KeyError(sha256) from None
                return obj["Body"].read()

            def exists(self, sha256):
                try:
                    self.s3.head_object(Bucket=self.bucket, Key=self.prefix + sha256)
                    return True
                except Exception:
                    return False

    ``put`` must be idempotent: the same bytes are often stored twice.
    ``get`` raises ``KeyError`` for an unknown hash.
    """

    def put(self, sha256: str, data: bytes) -> None: ...
    def get(self, sha256: str) -> bytes: ...
    def exists(self, sha256: str) -> bool: ...


class MemoryBlobStore:
    """Blobs in a dict — for tests, inline runs, and backends without a store."""

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    def put(self, sha256: str, data: bytes) -> None:
        self._blobs.setdefault(sha256, bytes(data))

    def get(self, sha256: str) -> bytes:
        return self._blobs[sha256]

    def exists(self, sha256: str) -> bool:
        return sha256 in self._blobs


class SQLiteBlobStore:
    """Blobs in a ``blobs`` table, by default in the run database itself."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._ready = False

    def _conn(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path)
        if not self._ready:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS blobs ("
                " sha256 TEXT PRIMARY KEY,"
                " size   INTEGER NOT NULL,"
                " data   BLOB    NOT NULL)"
            )
            conn.commit()
            self._ready = True
        return conn

    def put(self, sha256: str, data: bytes) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO blobs (sha256, size, data) VALUES (?, ?, ?)",
                (sha256, len(data), sqlite3.Binary(data)),
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, sha256: str) -> bytes:
        conn = self._conn()
        try:
            row = conn.execute("SELECT data FROM blobs WHERE sha256=?", (sha256,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(sha256)
        return bytes(row[0])

    def exists(self, sha256: str) -> bool:
        conn = self._conn()
        try:
            row = conn.execute("SELECT 1 FROM blobs WHERE sha256=?", (sha256,)).fetchone()
        finally:
            conn.close()
        return row is not None


def blob_store_for(storage: Any) -> BlobStore | None:
    """The blob store a storage backend provides, or None if it has none.

    A backend opts in with a ``blob_store()`` method.  Backends written before
    files existed simply lack it; the executor then keeps a run's files in
    memory, which works within a run but not across cached runs.
    """
    provider = getattr(storage, "blob_store", None)
    if provider is None:
        return None
    return provider() if callable(provider) else provider


# ---------------------------------------------------------------------------
# Names and hashing
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(name: str) -> str:
    """Return *name* if it is a plain file name, else raise FileOutputError.

    A plain name has no directory part, no control characters, and is not
    "." or "..", so writing it under a directory can never escape it.
    """
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in name)
        or len(name) > 255
    ):
        raise FileOutputError(f"Unsafe file name: {name!r}")
    return name


def safe_relpath(path: str) -> str:
    """Return *path* if it is a safe relative POSIX path inside a directory."""
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
        raise FileOutputError(f"Unsafe path in directory: {path!r}")
    for part in path.split("/"):
        safe_name(part)
    return path


def _guess_mime(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def _check_mime(mime: str) -> str:
    if not isinstance(mime, str) or not _MIME.fullmatch(mime):
        raise FileOutputError(f"Invalid MIME type: {mime!r}")
    return mime


def _check_sha(sha: str) -> str:
    if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
        raise FileOutputError(f"Invalid sha256: {sha!r}")
    return sha


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"  # pragma: no cover


_MATERIALIZE_ROOT: Path | None = None


def _materialize_root() -> Path:
    """A private (0700) directory for this process's on-disk copies.

    Private rather than a fixed path under /tmp, so no other user can plant a
    file where pbt expects to find a model's output.
    """
    global _MATERIALIZE_ROOT
    if _MATERIALIZE_ROOT is None:
        _MATERIALIZE_ROOT = Path(tempfile.mkdtemp(prefix="pbt-files-"))
    return _MATERIALIZE_ROOT


# ---------------------------------------------------------------------------
# File, Dir, Output
# ---------------------------------------------------------------------------

FileSource = Union[bytes, bytearray, str, "os.PathLike[str]", "IO[bytes]"]


class File:
    """One file produced by a model, or read from an upstream model's output.

    Construct one from bytes, a path, or a binary file object::

        pbt.File(png_bytes, name="logo.png")
        pbt.File("out/report.pdf")                 # name taken from the path
        pbt.File(io.BytesIO(data), name="a.zip", mime="application/zip")

    ``str(file)`` is a one-line handle — what ``{{ ref('m') }}`` puts in a
    prompt.  The bytes are read with :meth:`read_bytes`, :meth:`open` or
    :attr:`text`, or materialised on disk at :attr:`path`.
    """

    __slots__ = ("name", "mime", "size", "sha256", "_data", "_store")

    def __init__(
        self,
        data: FileSource,
        name: str | None = None,
        mime: str | None = None,
    ) -> None:
        if isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
            default_name = "file"
        elif isinstance(data, (str, os.PathLike)):
            path = Path(data)
            raw = path.read_bytes()
            default_name = path.name
        elif hasattr(data, "read"):
            raw = data.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            default_name = Path(getattr(data, "name", "") or "file").name
        else:
            raise TypeError(
                f"pbt.File() takes bytes, a path, or a binary file object; "
                f"got {type(data).__name__}."
            )
        self.name = safe_name(name or default_name)
        self.mime = _check_mime(mime or _guess_mime(self.name))
        self.size = len(raw)
        self.sha256 = _sha256(raw)
        self._data: bytes | None = raw
        self._store: BlobStore | None = None

    @classmethod
    def _stored(cls, sha256: str, name: str, mime: str, size: int, store: BlobStore) -> "File":
        obj = cls.__new__(cls)
        obj.sha256 = _check_sha(sha256)
        obj.name = safe_name(name)
        obj.mime = _check_mime(mime)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise FileOutputError(f"Invalid file size: {size!r}")
        obj.size = size
        obj._data = None
        obj._store = store
        return obj

    # -- reading --------------------------------------------------------------

    def read_bytes(self) -> bytes:
        """The file's bytes, verified against its sha256."""
        if self._data is not None:
            return self._data
        if self._store is None:
            raise FileOutputError(f"File '{self.name}' has no bytes and no blob store.")
        try:
            data = self._store.get(self.sha256)
        except KeyError:
            raise BlobMissingError(
                f"Blob {self.sha256} for file '{self.name}' is missing from the blob store."
            ) from None
        if _sha256(data) != self.sha256:
            raise FileOutputError(
                f"Blob {self.sha256} for file '{self.name}' does not match its hash."
            )
        return data

    @property
    def text(self) -> str:
        """The file decoded as UTF-8 (undecodable bytes replaced)."""
        return self.read_bytes().decode("utf-8", errors="replace")

    def open(self) -> "NamedBytesIO":
        """A fresh binary file object over the bytes, with ``.name`` set."""
        return NamedBytesIO(self.read_bytes(), self.name, sha256=self.sha256, mime=self.mime)

    @property
    def path(self) -> Path:
        """A read-only copy on disk, materialised on first access."""
        target = _materialize_root() / self.sha256 / self.name
        if not target.exists():
            self.save(target.parent)
            target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return target

    def save(self, directory: str | Path, name: str | None = None) -> Path:
        """Write the file into *directory* and return its path."""
        target = Path(directory) / safe_name(name or self.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.chmod(stat.S_IWUSR | stat.S_IRUSR)
        target.write_bytes(self.read_bytes())
        return target

    @property
    def is_image(self) -> bool:
        return self.mime.startswith("image/")

    # -- storage --------------------------------------------------------------

    def _persist(self, store: BlobStore) -> None:
        if self._data is not None:
            store.put(self.sha256, self._data)
            self._data = None
            self._store = store
        elif self._store is not store and self._store is not None:
            store.put(self.sha256, self.read_bytes())
            self._store = store

    def _manifest(self) -> dict:
        return {
            MARKER: "file",
            "sha256": self.sha256,
            "name": self.name,
            "mime": self.mime,
            "size": self.size,
        }

    # -- presentation ---------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"[file: {self.name} ({self.mime}, {_human_size(self.size)}, "
            f"sha256:{self.sha256[:12]})]"
        )

    def __repr__(self) -> str:
        return f"File(name={self.name!r}, mime={self.mime!r}, size={self.size}, sha256={self.sha256[:12]!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, File) and (self.sha256, self.name, self.mime) == (
            other.sha256, other.name, other.mime,
        )

    def __hash__(self) -> int:
        return hash((self.sha256, self.name, self.mime))


class Dir:
    """A directory of files, stored as one tree.

    ::

        pbt.Dir("build/site")                                   # from disk
        pbt.Dir({"index.html": html, "css/site.css": css}, name="site")

    Its :attr:`sha256` is a tree hash over each relative path and file hash, so
    renaming or editing any file changes it and nothing else does.  Symlinks
    are refused when reading from disk.
    """

    __slots__ = ("name", "entries", "sha256")

    def __init__(
        self,
        source: "str | os.PathLike[str] | dict[str, bytes | File]",
        name: str | None = None,
    ) -> None:
        entries: dict[str, File] = {}
        if isinstance(source, dict):
            for rel, value in source.items():
                rel = safe_relpath(str(PurePosixPath(rel)))
                leaf = PurePosixPath(rel).name
                if isinstance(value, File):
                    entries[rel] = value if value.name == leaf else _renamed(value, leaf)
                else:
                    entries[rel] = File(value, name=leaf)
            default_name = "dir"
        elif isinstance(source, (str, os.PathLike)):
            root = Path(source)
            if not root.is_dir():
                raise FileOutputError(f"pbt.Dir(): '{root}' is not a directory.")
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise FileOutputError(f"pbt.Dir(): refusing symlink '{path}'.")
                if path.is_file():
                    rel = safe_relpath(path.relative_to(root).as_posix())
                    entries[rel] = File(path)
            default_name = root.resolve().name or "dir"
        else:
            raise TypeError(
                f"pbt.Dir() takes a directory path or a dict of relative path → "
                f"bytes; got {type(source).__name__}."
            )
        self.name = safe_name(name or default_name)
        self.entries = dict(sorted(entries.items()))
        self.sha256 = _tree_hash(self.entries)

    @classmethod
    def _stored(cls, name: str, entries: dict[str, File], sha256: str) -> "Dir":
        obj = cls.__new__(cls)
        obj.name = safe_name(name)
        obj.entries = dict(sorted(entries.items()))
        obj.sha256 = _check_sha(sha256)
        if _tree_hash(obj.entries) != obj.sha256:
            raise FileOutputError(f"Directory '{name}' does not match its tree hash.")
        return obj

    def files(self) -> list[File]:
        return list(self.entries.values())

    def __iter__(self) -> Iterator[File]:
        return iter(self.entries.values())

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, relpath: str) -> File:
        return self.entries[relpath]

    @property
    def size(self) -> int:
        return sum(f.size for f in self.entries.values())

    @property
    def path(self) -> Path:
        """A copy of the tree on disk, materialised on first access."""
        target = _materialize_root() / self.sha256 / self.name
        if not target.exists():
            self.save(target.parent)
        return target

    def save(self, directory: str | Path) -> Path:
        """Write the tree to ``directory / self.name`` and return that path."""
        root = Path(directory) / self.name
        for rel, file in self.entries.items():
            parent = root.joinpath(*PurePosixPath(rel).parts[:-1])
            file.save(parent)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _persist(self, store: BlobStore) -> None:
        for file in self.entries.values():
            file._persist(store)

    def _manifest(self) -> dict:
        return {
            MARKER: "dir",
            "sha256": self.sha256,
            "name": self.name,
            "entries": {
                rel: {"sha256": f.sha256, "mime": f.mime, "size": f.size}
                for rel, f in self.entries.items()
            },
        }

    def __str__(self) -> str:
        count = len(self.entries)
        lines = [
            f"[dir: {self.name} ({count} file{'s' if count != 1 else ''}, "
            f"{_human_size(self.size)}, sha256:{self.sha256[:12]})]"
        ]
        for rel, f in list(self.entries.items())[:_LISTING_LIMIT]:
            lines.append(f"  - {rel} ({f.mime}, {_human_size(f.size)})")
        if count > _LISTING_LIMIT:
            lines.append(f"  - … {count - _LISTING_LIMIT} more")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Dir(name={self.name!r}, files={len(self.entries)}, sha256={self.sha256[:12]!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Dir) and (self.sha256, self.name) == (other.sha256, other.name)

    def __hash__(self) -> int:
        return hash((self.sha256, self.name))


class Output:
    """Text plus files — the common shape of a multimodal model response.

    ::

        return pbt.Output("Here is the logo you asked for.", files=[pbt.File(png, name="logo.png")])

    ``str(output)`` is the text followed by one handle per file, so
    ``{{ ref('m') }}`` keeps working in templates written for text models.
    Individual files are ``output.files["logo.png"]``.
    """

    __slots__ = ("text", "files")

    def __init__(self, text: str = "", files: "list[File | Dir] | dict[str, File | Dir] | None" = None) -> None:
        if not isinstance(text, str):
            raise TypeError(f"pbt.Output text must be a str; got {type(text).__name__}.")
        self.text = text
        items = list(files.values()) if isinstance(files, dict) else list(files or [])
        self.files: dict[str, File | Dir] = {}
        for item in items:
            if not isinstance(item, (File, Dir)):
                raise TypeError(
                    f"pbt.Output files must be pbt.File or pbt.Dir; got {type(item).__name__}."
                )
            if item.name in self.files:
                raise FileOutputError(f"pbt.Output has two files named '{item.name}'.")
            self.files[item.name] = item

    def _persist(self, store: BlobStore) -> None:
        for item in self.files.values():
            item._persist(store)

    def __str__(self) -> str:
        handles = "\n".join(str(item) for item in self.files.values())
        if not handles:
            return self.text
        return f"{self.text}\n\n{handles}" if self.text else handles

    def __repr__(self) -> str:
        return f"Output(text={self.text[:40]!r}, files={list(self.files)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Output) and (self.text, self.files) == (other.text, other.files)


def _renamed(file: File, name: str) -> File:
    copy = File.__new__(File)
    copy.name = safe_name(name)
    copy.mime = file.mime
    copy.size = file.size
    copy.sha256 = file.sha256
    copy._data = file._data
    copy._store = file._store
    return copy


def _tree_hash(entries: dict[str, File]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(entries):
        digest.update(rel.encode("utf-8") + b"\x00" + entries[rel].sha256.encode("ascii") + b"\n")
    return digest.hexdigest()


class NamedBytesIO(io.BytesIO):
    """A BytesIO with the ``.name`` LLM clients read to guess a MIME type.

    Also carries ``.sha256`` and ``.mime``, so the prompt cache can key on the
    hash without re-reading the bytes.
    """

    def __init__(self, data: bytes, name: str, sha256: str | None = None, mime: str | None = None) -> None:
        super().__init__(data)
        self.name = name
        self.sha256 = sha256
        self.mime = mime


# ---------------------------------------------------------------------------
# Walking values
# ---------------------------------------------------------------------------

_RICH = (File, Dir, Output)


def contains_files(value: Any) -> bool:
    """True when *value* is, or holds anywhere inside it, a File/Dir/Output."""
    if isinstance(value, _RICH):
        return True
    if isinstance(value, dict):
        return any(contains_files(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_files(v) for v in value)
    return False


def iter_files(value: Any, prefix: str = "") -> Iterator[tuple[str, File]]:
    """Yield ``(label, file)`` for every file in *value*, directories expanded.

    The label is where the file sits: ``image`` for a dict key, ``0`` for a
    list item, ``site/css/app.css`` for a file inside a directory.
    """
    def join(label: str) -> str:
        return f"{prefix}.{label}" if prefix else label

    if isinstance(value, File):
        yield (prefix or value.name), value
    elif isinstance(value, Dir):
        base = prefix or value.name
        for rel, f in value.entries.items():
            yield f"{base}/{rel}", f
    elif isinstance(value, Output):
        for name, item in value.files.items():
            yield from iter_files(item, join(name) if prefix else name)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_files(item, join(str(key)))
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            yield from iter_files(item, join(str(idx)))


def persist_files(value: Any, store: BlobStore) -> None:
    """Write the bytes of every file in *value* to *store*."""
    if isinstance(value, _RICH):
        value._persist(store)
    elif isinstance(value, dict):
        for item in value.values():
            persist_files(item, store)
    elif isinstance(value, (list, tuple)):
        for item in value:
            persist_files(item, store)


def display_text(value: Any) -> str:
    """The textual part of an output — what a text-only reader should see."""
    if isinstance(value, Output):
        return value.text
    if isinstance(value, str):
        return value
    if isinstance(value, (File, Dir)):
        return ""
    return json.dumps(to_jsonable(value), indent=2)


def to_jsonable(value: Any) -> Any:
    """*value* as plain JSON data, each file replaced by its manifest dict.

    For APIs and reports, not storage: the result is not escaped and must
    never be fed back into :func:`decode_output`.
    """
    if isinstance(value, (File, Dir)):
        manifest = value._manifest()
        manifest["type"] = manifest.pop(MARKER)
        return manifest
    if isinstance(value, Output):
        return {"type": "output", "text": value.text,
                "files": {k: to_jsonable(v) for k, v in value.files.items()}}
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Encoding for storage
# ---------------------------------------------------------------------------

def encode_output(value: Any) -> str:
    """Encode an output value as the string storage keeps.

    Text stays text and file-free JSON values stay plain JSON, exactly as
    before files existed.  A value with files becomes the ``$pbt:v1``
    envelope — the one place a ``"$pbt"`` marker is ever written.  The files
    must already be persisted (see :func:`persist_files`).
    """
    if isinstance(value, str):
        return _TEXT_ESCAPE_PREFIX + value if value.startswith("$pbt:") else value
    if not contains_files(value):
        return json.dumps(value)
    return ENVELOPE_PREFIX + json.dumps(_to_wire(value), sort_keys=True)


def _to_wire(value: Any) -> Any:
    if isinstance(value, (File, Dir)):
        if isinstance(value, File) and value._store is None:
            raise FileOutputError(f"File '{value.name}' was encoded before being stored.")
        return value._manifest()
    if isinstance(value, Output):
        return {
            MARKER: "output",
            "text": value.text,
            "files": {name: _to_wire(item) for name, item in value.files.items()},
        }
    if isinstance(value, dict):
        wired = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FileOutputError(f"Output dict keys must be strings; got {key!r}.")
            wired["$" + key if _MARKER_KEY.fullmatch(key) else key] = _to_wire(item)
        return wired
    if isinstance(value, (list, tuple)):
        return [_to_wire(item) for item in value]
    return value


def decode_output(raw: str | None, store: BlobStore | None, check_blobs: bool = False) -> Any:
    """Invert :func:`encode_output`: a stored string back to an output value.

    Plain text and plain JSON come back as the same string they were (JSON is
    parsed by the executor, as it always has been).  An envelope comes back
    as File/Dir/Output objects bound to *store*.

    With *check_blobs*, every referenced blob must exist in *store*, else
    :class:`BlobMissingError` — the prompt cache uses this so a hit whose
    bytes were deleted is treated as a miss rather than served broken.
    """
    if raw is None:
        return None
    if raw.startswith(_TEXT_ESCAPE_PREFIX):
        return raw[len(_TEXT_ESCAPE_PREFIX):]
    if not raw.startswith(ENVELOPE_PREFIX):
        return raw
    if store is None:
        raise FileOutputError("This output holds files but no blob store is available.")
    try:
        data = json.loads(raw[len(ENVELOPE_PREFIX):])
    except json.JSONDecodeError as exc:
        raise FileOutputError(f"Corrupt stored output: {exc}") from exc
    value = _from_wire(data, store)
    if check_blobs:
        for _, file in iter_files(value):
            if not store.exists(file.sha256):
                raise BlobMissingError(f"Blob {file.sha256} for '{file.name}' is missing.")
    return value


def _from_wire(data: Any, store: BlobStore) -> Any:
    if isinstance(data, list):
        return [_from_wire(item, store) for item in data]
    if not isinstance(data, dict):
        return data
    if MARKER in data:
        return _from_marker(data, store)
    return {
        (key[1:] if _ESCAPED_MARKER_KEY.fullmatch(key) else key): _from_wire(item, store)
        for key, item in data.items()
    }


def _from_marker(data: dict, store: BlobStore) -> Any:
    kind = data.get(MARKER)
    if kind == "file":
        return File._stored(data.get("sha256"), data.get("name"), data.get("mime"), data.get("size"), store)
    if kind == "dir":
        raw_entries = data.get("entries")
        if not isinstance(raw_entries, dict):
            raise FileOutputError("Stored directory has no entries.")
        entries = {}
        for rel, meta in raw_entries.items():
            rel = safe_relpath(rel)
            if not isinstance(meta, dict):
                raise FileOutputError(f"Stored directory entry {rel!r} is malformed.")
            entries[rel] = File._stored(
                meta.get("sha256"), PurePosixPath(rel).name, meta.get("mime"), meta.get("size"), store,
            )
        return Dir._stored(data.get("name"), entries, data.get("sha256"))
    if kind == "output":
        text = data.get("text", "")
        files = data.get("files") or {}
        if not isinstance(text, str) or not isinstance(files, dict):
            raise FileOutputError("Stored output is malformed.")
        items = [_from_wire(item, store) for item in files.values()]
        if not all(isinstance(item, (File, Dir)) for item in items):
            raise FileOutputError("Stored output lists something that is not a file.")
        return Output(text, files=items)
    raise FileOutputError(f"Unknown stored object type: {kind!r}")


# ---------------------------------------------------------------------------
# Resolving a promptfile name against upstream outputs
# ---------------------------------------------------------------------------

def split_model_path(name: str, model_names: "set[str] | dict") -> tuple[str, list[str]] | None:
    """Split ``"logo.image"`` into ``("logo", ["image"])`` if ``logo`` is a model.

    The longest dotted prefix naming a model wins, so model names containing
    dots still work.  Returns None when no prefix names a model — the name is
    then a run-level promptfile.
    """
    parts = name.split(".")
    for cut in range(len(parts), 0, -1):
        head = ".".join(parts[:cut])
        if head in model_names:
            return head, parts[cut:]
    return None


def select_path(value: Any, path: list[str], label: str) -> Any:
    """Follow *path* (dict keys, list indices, Output/Dir members) into *value*."""
    for step in path:
        if isinstance(value, Output):
            value = value.files.get(step, _MISSING)
        elif isinstance(value, Dir):
            value = value.entries.get(step, _MISSING)
        elif isinstance(value, dict):
            value = value.get(step, _MISSING)
        elif isinstance(value, list) and step.isdigit() and int(step) < len(value):
            value = value[int(step)]
        else:
            value = _MISSING
        if value is _MISSING:
            raise FileOutputError(f"promptfile '{label}': '{step}' not found in the upstream output.")
    return value


_MISSING = object()


# ---------------------------------------------------------------------------
# Exporting
# ---------------------------------------------------------------------------

def export_files(value: Any, directory: str | Path) -> list[Path]:
    """Write every file in *value* under *directory*; return the paths written.

    Files keep their own names, and files inside a :class:`Dir` their relative
    paths.  Two files with the same name are told apart by where they sit in
    the output (``image-logo.png``).  Nothing is written outside *directory*.
    """
    root = Path(directory)
    written: list[Path] = []
    taken: set[str] = set()
    for label, file in iter_files(value):
        if "/" in label:  # inside a Dir: keep the tree
            rel = safe_relpath(label)
        else:
            rel = file.name
            if rel in taken:
                rel = safe_name(f"{label.replace('.', '-')}-{file.name}")
        taken.add(rel)
        parts = PurePosixPath(rel).parts
        written.append(file.save(root.joinpath(*parts[:-1]), parts[-1]))
    return written
