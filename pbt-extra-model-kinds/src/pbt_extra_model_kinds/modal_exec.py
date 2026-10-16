"""
A ``python`` bullet type: run the bullet's code in a Modal Sandbox.

``template`` needs no model type of its own — pbt ships one. Python cannot be
had that cheaply: the code has to run somewhere that is not this process, and
pbt's built-in ``execute_python`` runs it in-process. So this registers a model
type of its own.

A python bullet holds no code a person typed. It runs what its one child
produced — an LLM writes a script, the bullet above it executes that script —
so the only thing that ever reaches the sandbox came from upstream.

pbt exposes a public model-kind registry, so this is a plain
``pbt.ModelKind`` registered under ``python_modal``. The executor owns
rendering, the prompt cache, JSON parsing and storage; the exec function below
only supplies the interesting middle — hand the upstream output to a sandbox
and return what it printed.

Auth is Modal's usual environment: ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET``
(or a ``modal token new`` profile on the box). Nothing is read from the browser
— sandboxes cost money, so the key stays server-side.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tomllib
from typing import Any

import pbt

MODEL_TYPE = "python_modal"

# The config key carrying this bullet's extra packages, comma-separated.
PACKAGES_KEY = "packages"

# The variable holding the boilerplate for a prompt that asks for a script. Its
# value is written by the server (see `standard_instructions`), not by a person:
# it names the packages *this* sandbox actually has and the exact syntax that
# gets more, both of which the app knows and the person writing the prompt
# would otherwise have to keep in step by hand.
INSTRUCTIONS_VAR = "coding_instructions"

# The bullet is prepended with this, exactly as template bullets get their own
# config line. See `main.py:_build_source`.
CONFIG_LINE = '{{ config(model_type="%s") }}' % MODEL_TYPE


def config_line(extra: str = "") -> str:
    """The bullet's config line, carrying any packages the bullet asked for.

    They travel *in the model source* rather than in a module-level variable
    because the server answers requests concurrently: a global would let one
    run's packages end up in another's sandbox. `spec.config` belongs to one
    model, which is exactly the scope this needs.
    """
    if not extra.strip():
        return CONFIG_LINE
    return '{{ config(model_type="%s", %s="%s") }}' % (
        MODEL_TYPE,
        PACKAGES_KEY,
        extra.replace('"', "").strip(),
    )

APP_NAME = os.environ.get("MODAL_APP_NAME", "mindmap-python")

# Always present in the sandbox, so the common case pays no image build.
BASE_PACKAGES = ["numpy", "pandas", "requests"]

# More, for one deployment: MODAL_PACKAGES="scipy, pillow==11.*, beautifulsoup4".
#
# Server-side, and only server-side. A python bullet runs code written upstream
# — by an LLM, most of the time — so a *bullet* asking for a package would mean
# the package name was chosen by the model too, and pip runs a package's own
# build code before the script gets to run. Whoever deploys the server picks the
# image; nothing that arrives over HTTP can add to it.
PACKAGES_ENV = "MODAL_PACKAGES"

# A requirement as `uv pip install` should see it: a name, optional extras,
# optional version pin. Anything else — an index URL, a flag, a path, a git
# reference — is refused rather than passed through as an argument.
_REQUIREMENT = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*"          # name
    r"(\[[A-Za-z0-9,._-]+\])?"              # [extras]
    r"((==|>=|<=|~=|!=|<|>)[A-Za-z0-9._*+!-]+)?$"  # ==version
)


def _split_env() -> tuple[list[str], list[str]]:
    """The deployment's extras as ``(accepted, refused)``."""
    return split_requirements(os.environ.get(PACKAGES_ENV, ""))


def packages() -> list[str]:
    """What every sandbox installs: the base list plus this server's extras."""
    extra, _ = _split_env()
    return BASE_PACKAGES + [p for p in extra if p not in BASE_PACKAGES]


# PEP 723 — inline script metadata. The accepted way for a single-file script to
# state what it needs, and what `uv run` and `pipx run` read:
#
#     # /// script
#     # dependencies = ["httpx", "rich"]
#     # ///
#
# Using the standard rather than inventing a marker means a script written here
# runs unchanged anywhere else, and a model asked for "a PEP 723 header" already
# knows what that is.
_PEP723 = re.compile(
    r"(?m)^# /// script\s*$\n(?P<body>(?:^#(?: .*)?$\n)*?)^# ///\s*$",
)


def script_requirements(source: str) -> tuple[list[str], list[str]]:
    """What a script asks for in its PEP 723 block, as ``(accepted, refused)``.

    A script with no block, or an unparseable one, asks for nothing — a comment
    that is not valid metadata is a comment, and refusing to run over it would
    be worse than ignoring it.
    """
    match = _PEP723.search(source)
    if not match:
        return [], []
    body = "\n".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in match.group("body").splitlines()
    )
    try:
        declared = tomllib.loads(body).get("dependencies", [])
    except (tomllib.TOMLDecodeError, AttributeError):
        return [], []
    if not isinstance(declared, list):
        return [], []
    return split_requirements(" ".join(str(d) for d in declared if isinstance(d, str)))


def standard_instructions() -> str:
    """What to tell a model that is being asked to write a script for a sandbox.

    Three things it cannot know: that its answer is executed rather than read,
    what is already installed, and how to ask for more. The package list is the
    real one for this server, so the instructions cannot drift from
    the sandbox the way a hand-written paragraph would.
    """
    have = packages()
    return (
        "Write Python only — the file is run exactly as you write it, so no "
        "explanation outside comments. A ``` fence around it is fine.\n"
        f"These are already installed: {', '.join(have)}.\n"
        "For anything else, declare it in a PEP 723 block at the top of the "
        "file and it will be installed before the script runs:\n"
        "\n"
        "# /// script\n"
        '# dependencies = ["httpx", "rich"]\n'
        "# ///"
    )


def _bare(requirement: str) -> str:
    """A requirement's name, without extras or a version pin."""
    return re.split(r"[\[<>=!~]", requirement, 1)[0].strip().lower()


def split_requirements(raw: str) -> tuple[list[str], list[str]]:
    """A comma- or space-separated list as ``(accepted, refused)``.

    The same rule as `MODAL_PACKAGES`: a name, optional extras, optional pin,
    and nothing that could be read as a flag or an index URL.
    """
    good: list[str] = []
    bad: list[str] = []
    for item in re.split(r"[,\s]+", (raw or "").strip()):
        if not item:
            continue
        target = good if _REQUIREMENT.match(item) else bad
        if item not in target:
            target.append(item)
    return good, bad


def rejected() -> list[str]:
    """Entries in `MODAL_PACKAGES` that are not requirements — a typo, usually.

    Reported rather than dropped: a package the person meant to have and does
    not shows up in the sandbox as an ImportError from a script they did not
    write, which is a long way from the environment variable that caused it.
    """
    return _split_env()[1]

TIMEOUT_SECONDS = int(os.environ.get("MODAL_TIMEOUT_SECONDS", "60"))
CPU = 1
MEMORY_MB = 2048

# Sandbox output is a bullet's result and travels back to the browser; cap it.
MAX_OUTPUT_CHARS = 10_000

_app = None
_base_image = None
# Images keyed by the extra packages that went into them. Modal caches a built
# image, but building the definition again per run would still ask it to, so a
# graph with the same packages reuses the same object run after run.
_images: dict[tuple[str, ...], object] = {}


def enabled() -> bool:
    """Whether Modal credentials are present. The UI hides `python` if not."""
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    # A `modal token new` profile on the machine works just as well.
    return os.path.exists(os.path.expanduser("~/.modal.toml"))


def _lookup():
    """The Modal app and base image, created once and reused.

    Deferred rather than done at import: both calls hit the network, and a
    server with no Modal credentials must still start and serve prompt bullets.

    Held for the life of the process, so a change to `MODAL_PACKAGES` needs a
    restart — and costs one image build on the next run, after which Modal has
    it cached.
    """
    global _app, _base_image
    if _app is None:
        import modal

        _app = modal.App.lookup(APP_NAME, create_if_missing=True)
        _base_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
            *packages()
        )
    return _app, _base_image


_FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)


def _unfence(text: str) -> str:
    """The runnable code inside *text*: the first fenced block, or all of it.

    An empty python bullet runs its inputs, and those inputs usually come from
    an LLM that was asked for a script — which fences it, and often wraps it in
    "Here is a simple script…" and a How-to-run section, "no commentary" or
    not. Neither prose nor a fence is valid Python, so when there is a fenced
    block, that block is the program. The *first* one, not all of them: a
    chatty answer tends to follow the real script with variations on it.
    """
    m = _FENCE.search(text)
    return m.group(1) if m else text


def _inherited_code(inputs: list[Any]) -> str:
    """The program: whatever this bullet's child produced.

    A python bullet never carries code of its own. It is an operator — one
    child writes a script, the python bullet above it runs that script — so
    there is no text of its own to prefer, and nothing a person typed on the
    bullet can reach the sandbox. `main.py` holds it to a single input, so the
    join below is over a list of one; it is written this way so a graph that
    somehow arrives with more still runs rather than picking one at random.
    """
    return "\n".join(_unfence(str(i)) for i in inputs if str(i).strip())


def _program(inputs: list[Any], code: str) -> str:
    """The full program to run in the sandbox: the inputs, then the code.

    The sandbox is a different machine, so `ref()` cannot reach back into this
    process's `model_outputs` the way pbt's in-process handler does. The
    outputs are serialised into the source instead — through `json.dumps`
    twice, so each payload arrives as a Python string literal that no quote or
    newline in an LLM's answer can break out of.

    The bullet's own code is passed the same way and `compile`d rather than
    pasted in below the header, so a traceback reports the line the person
    wrote — line 1 of the bullet is line 1 in the error.
    """
    return (
        "import json as _json\n"
        f"inputs = _json.loads({json.dumps(json.dumps(inputs))})\n"
        "def ref(i=0):\n"
        "    return inputs[i]\n"
        f"exec(compile(_json.loads({json.dumps(json.dumps(code))}), "
        '"<bullet>", "exec"), globals())\n'
    )


def _image_for(extra: list[str]):
    """The sandbox image: the server's packages, plus this bullet's.

    The first run with a new set pays for the build; Modal has it cached after
    that, and so does `_images` within this process.
    """
    _, base_image = _lookup()
    if not extra:
        return base_image
    key = tuple(sorted(extra))
    if key not in _images:
        _images[key] = base_image.uv_pip_install(*key)
    return _images[key]


def _run_sandbox(code: str, extra: list[str]) -> str:
    """Execute *code* in a fresh sandbox. Blocking — call it off the loop."""
    import modal

    app, _ = _lookup()

    sb = modal.Sandbox.create(
        image=_image_for(extra),
        app=app,
        timeout=TIMEOUT_SECONDS,
        cpu=CPU,
        memory=MEMORY_MB,
    )
    try:
        p = sb.exec("python", "-c", code)
        out = p.stdout.read()
        err = p.stderr.read()
        returncode = p.wait()
    finally:
        sb.terminate()

    if returncode:
        # A traceback is the useful part of a failed run, so put it in the
        # error rather than returning a bullet result that looks like success.
        detail = (err or out or "").strip()[-MAX_OUTPUT_CHARS:]
        raise RuntimeError(f"Python bullet exited {returncode}:\n{detail}")

    result = (out or "").strip()
    if not result and err:
        result = err.strip()
    return result[:MAX_OUTPUT_CHARS]


async def _run(inputs: list[Any], extra: str = "") -> str:
    if not enabled():
        raise RuntimeError(
            "Python bullets run on Modal, which is not configured on this "
            "server. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET."
        )

    if refused := rejected():
        raise RuntimeError(
            f"{PACKAGES_ENV} lists something that is not a package requirement: "
            f"{', '.join(refused)}. Running with it quietly missing would surface "
            f"as an ImportError from a script nobody wrote, so nothing is run."
        )

    wanted, refused = split_requirements(extra)
    if refused:
        raise RuntimeError(
            f"This bullet's packages include something that is not a package "
            f"requirement: {', '.join(refused)}. A name, optional [extras] and "
            f"an optional version pin — nothing that reads as a flag or a URL."
        )

    source = _inherited_code(inputs)
    if not source.strip():
        raise RuntimeError(
            "A python bullet runs the code its child produces, and this "
            "one received nothing to run. Give it a child that outputs a "
            "script."
        )

    # What the script itself asked for, on top of what the run offered it.
    declared, declined = script_requirements(source)
    if declined:
        raise RuntimeError(
            f"The script's PEP 723 block asks for something that is not a "
            f"package requirement: {', '.join(declined)}."
        )
    for name in declared:
        if name not in wanted:
            wanted.append(name)

    # Anything the image already has is dropped rather than reinstalled: an
    # extra name means a different image, and a different image means a build,
    # for a package that was there all along. `requests` typed into the
    # variables table is the ordinary way to hit this.
    have = {_bare(p) for p in packages()}
    wanted = [w for w in wanted if _bare(w) not in have]

    # Modal's client is blocking, and pbt runs independent branches
    # concurrently — keep one sandbox from stalling the others.
    return await asyncio.to_thread(_run_sandbox, _program(inputs, source), wanted)


# Registered on import: the decorator runs when this module body executes, so
# importing `modal_exec` is what teaches pbt the kind. The rendered template is
# Python source, so prose prepended to it would not compile — hence
# `accepts_global_instruction=False`.
@pbt.model_kind(MODEL_TYPE, config_keys={PACKAGES_KEY}, accepts_global_instruction=False)
async def execute(rendered: str, call: pbt.ModelCall) -> str:
    """Run the code this bullet's child produced, in a Modal Sandbox (gVisor).

    The bullet carries no code of its own — it is an operator, not an editor.
    Its child's output is the program, and is also readable by that program as
    ``inputs`` (a list, or ``ref(0)``). Whatever it prints is the bullet's
    output, which flows on downstream.
    """
    inputs = [call.outputs.get(dep) for dep in call.spec.depends_on]
    extra = str(call.spec.config.get(PACKAGES_KEY, ""))

    # A python bullet's own source renders to nothing (its refs live in a
    # Jinja comment — see `main.py:_python_source`), so the upstream outputs
    # *are* the program and have to be in the cache key. They are appended to
    # the text handed to `call.compute`, which is only ever used to build that
    # key; the prompt shown in the run report was recorded when the executor
    # rendered.
    # The packages go in the key explicitly: `{{ config(...) }}` renders to
    # nothing, so the environment a script ran against is invisible in
    # `rendered` — and the same script against a different environment is a
    # different run, which may well answer differently.
    cache_text = (
        rendered + "\x00" + json.dumps(inputs, sort_keys=True, default=str) + "\x00" + extra
    )

    return await call.compute(cache_text, compute=lambda: _run(inputs, extra))
