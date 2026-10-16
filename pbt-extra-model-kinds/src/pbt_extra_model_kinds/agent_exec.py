"""
An ``agent`` bullet type: a coding agent in a Modal sandbox, with an optional MCP server.

A prompt bullet asks a model once. An agent bullet hands its rendered text — its
own words plus whatever flowed up from its children and `@` references — to
OpenCode as a task, in a fresh sandbox, and the agent works at it with its
tools (bash, file reads and edits, …) until it answers. That answer is the
bullet's output.

Any stdio MCP server can ride along. Everything lives in the one sandbox:
nothing is hosted and no ports are exposed.

    Modal sandbox       (main process: a plain ``sleep``)
    └── run_agent.py    writes OpenCode's config, checks the MCP server comes
                        up, runs ``opencode run`` headless and reads its events

OpenCode is an MCP client itself: it launches the server over stdio and holds
the one session for the whole run, which keeps the server's state alive across
steps, and hands the model its tools (and any images they return) directly.

The server to launch is a per-bullet setting — ``{{ config(model_type=
"agent_modal", mcp_server="uvx some-mcp-server") }}`` — so it travels in the
model source like a python bullet's packages do (see `modal_exec.config_line`).

The model's key does *not*: a key written into the source would end up in the
cache key and in every export. It is carried per run in a context variable set
by ``main.run`` (``use_run``), along with the run's step limit and where its
live log lines go,, which every task pbt spawns inherits, and
reaches the sandbox as a Modal secret built on the spot.

The bullet's output is a JSON object, not bare text:

    {"output": <the agent's answer>, "logs": [...], "run_time": <seconds>}

``logs`` is the run end to end, one line per event with its offset from the
start: the sandbox being created, the MCP server coming up and each tool call it
served, every agent step (its thought, its command, what the command returned),
how the agent finished, and the teardown. ``run_time`` is wall-clock seconds from
creating the sandbox to terminating it. When the bullet has JSON enforced,
``output`` is the answer parsed, and an answer that will not parse fails it.

A bullet set to produce files (``produces_files=True``) also has ``files``: the
agent is told to save them in ``/root/outputs/``, and each one it saved comes
back as a ``pbt.File`` — handed to the bullets downstream as attachments.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import pathlib
import re
import secrets as secrets_mod
import shlex
import time
from dataclasses import dataclass
from typing import Any, Callable

import pbt
from pbt.executor.run_context import parse_json_output as pbt_json

from . import modal_exec
from .models import ENV_KEYS, model_name

MODEL_TYPE = "agent_modal"

# The config key naming the MCP server command. Empty means none.
MCP_KEY = "mcp_server"
# The config key holding the bullet's own step limit. Absent means the run's.
STEPS_KEY = "steps"
# The config key saying the bullet must hand on files, not only text.
FILES_KEY = "produces_files"
# The config key holding the bullet's wait between model calls, in seconds.
BACKOFF_KEY = "backoff"
# The config key giving the agent a visible Chromium that a person can watch
# and take over, over noVNC on a Modal tunnel — see `_start_browser`.
BROWSER_KEY = "remote_browser"

# The remote browser: a virtual display, Chromium on it with the DevTools
# protocol open to the agent, and noVNC serving that display on a tunnel.
DISPLAY = ":99"
SCREEN = "1280x800x24"
CDP_PORT = 9222
VNC_PORT = 6080

# Where an agent asked for files saves them — see `run_agent.FILES_NOTE`.
OUTPUTS_DIR = "/root/outputs"
# What is collected from it. Files travel to every bullet downstream, so a run
# that saves more keeps the first ones and says so. Large ones reach the browser
# through the bucket, not the run response (see `main._file_info`), so the byte
# cap is about the server's memory, not the page's.
MAX_FILES = int(os.environ.get("AGENT_MAX_FILES", "10"))
MAX_FILES_BYTES = int(os.environ.get("AGENT_MAX_FILES_BYTES", str(100 * 1024 * 1024)))

APP_NAME = os.environ.get("AGENT_APP_NAME", "mindmap-agent")

# Whole-sandbox lifetime, which bounds the agent's run and the server's start.
TIMEOUT_SECONDS = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "900"))
# How long the MCP server may take to come up. A `uvx`/`npx` server downloads
# itself on first start, which is most of this.
MCP_START_SECONDS = int(os.environ.get("AGENT_MCP_START_SECONDS", "180"))
# Steps before the model is told to stop using tools and answer. The default;
# a run may ask for another within MAX_STEP_LIMIT (Settings → agent steps).
STEP_LIMIT = int(os.environ.get("AGENT_STEP_LIMIT", "30"))
MAX_STEP_LIMIT = int(os.environ.get("AGENT_MAX_STEP_LIMIT", "200"))
# The longest back-off a bullet may ask for between model calls, in seconds.
MAX_BACKOFF = float(os.environ.get("AGENT_MAX_BACKOFF_SECONDS", "300"))

# How `run_agent.py` flags a log event printed live — see its LIVE_MARKER.
LIVE_MARKER = "@@agent-event "
CPU = 2
MEMORY_MB = 4096

# The agent's answer is a bullet's result and travels back to the browser.
MAX_OUTPUT_CHARS = 20_000

# Deployment-only additions to the sandbox image, like `MODAL_PACKAGES` for
# python bullets: system libraries an MCP server needs (`libgl1` for a CAD
# server, say), Python packages, and Modal secrets holding the server's own API
# keys. Server-side and only server-side — nothing that arrives over HTTP picks
# what gets installed or which of your secrets a sandbox can read.
APT_ENV = "AGENT_APT_PACKAGES"
PIP_ENV = "AGENT_PIP_PACKAGES"
SECRETS_ENV = "AGENT_MODAL_SECRETS"

# An apt package name, per Debian policy. Anything else is refused rather than
# handed to `apt-get install` as an argument.
_APT_NAME = re.compile(r"^[a-z0-9][a-z0-9+.-]+$")

# Pinned: the bullet's log and answer are read from OpenCode's JSON event
# stream, whose shape is OpenCode's to change.
OPENCODE_VERSION = "1.18.32"

AGENT_DIR = pathlib.Path(__file__).resolve().parent / "agent"
TASK_PATH = "/tmp/agent_task.md"
RESULT_PATH = "/tmp/agent_result.json"

# The log travels back to the browser and on to any bullet downstream. A run
# that goes past this keeps its start and its end, which is where the story is.
MAX_LOG_LINES = 400

# OpenCode's name for each provider this server offers.
_OPENCODE_PROVIDER = {"gemini": "google", "openai": "openai", "anthropic": "anthropic", "openrouter": "openrouter"}
# The key variable OpenCode reads, where it differs from this server's ENV_KEYS.
_OPENCODE_KEY_ENV = {"gemini": "GOOGLE_GENERATIVE_AI_API_KEY"}

# A live log line for one bullet: (pbt model name, line).
LogSink = Callable[[str, str], None]


@dataclass(frozen=True)
class RunSettings:
    """What an agent bullet takes from the run it is part of."""

    provider: str = "gemini"
    api_key: str | None = None
    # Settings → `llm_model`; None means the server's default for the provider.
    model: str | None = None
    # Settings → agent steps; clamped by `steps_for`.
    steps: int = STEP_LIMIT
    # Where each log line goes the moment it happens, for the live run log.
    on_log: LogSink | None = None
    # Settings → the user's own Agent37 key, for preserve-disk bullets on their
    # own workspace; None runs them on the server's shared instance.
    agent37_key: str | None = None
    # Who is running: the signed-in email, else the browser's session id. An
    # Agent37 bullet runs in this user's own Hermes profile.
    user: str = ""


# The run in progress. A context variable rather than a global because the
# server answers requests concurrently: each `/run` sets its own, and the tasks
# pbt spawns for it (and the threads they start) inherit that one.
_settings: contextvars.ContextVar[RunSettings] = contextvars.ContextVar(
    "agent_settings", default=RunSettings()
)


def use_run(settings: RunSettings) -> contextvars.Token:
    """Bind this run's provider, key, step limit and log sink for any agent bullet in it."""
    return _settings.set(settings)


def reset_run(token: contextvars.Token) -> None:
    _settings.reset(token)


def steps_for(requested: int | None) -> int:
    """A step limit a run may ask for: the default when it names none."""
    if not requested or requested < 1:
        return STEP_LIMIT
    return min(int(requested), MAX_STEP_LIMIT)


def config_line(
    mcp_server: str = "",
    steps: int = 0,
    produces_files: bool = False,
    backoff: float = 0,
    remote_browser: bool = False,
) -> str:
    """The bullet's config line, naming its MCP server, step limit, whether
    it hands on files, its back-off and whether it has a remote browser, each
    only when set.

    The command goes through `json.dumps`, whose escapes Jinja's string
    literals read the same way, so no quote in it can end the literal early.
    """
    args = ['model_type="%s"' % MODEL_TYPE]
    command = " ".join(mcp_server.split())
    if command:
        args.append("%s=%s" % (MCP_KEY, json.dumps(command)))
    if steps and steps > 0:
        args.append("%s=%d" % (STEPS_KEY, int(steps)))
    if produces_files:
        args.append("%s=True" % FILES_KEY)
    if backoff and backoff > 0:
        args.append("%s=%g" % (BACKOFF_KEY, min(float(backoff), MAX_BACKOFF)))
    if remote_browser:
        args.append("%s=True" % BROWSER_KEY)
    return "{{ config(%s) }}" % ", ".join(args)


def enabled() -> bool:
    """Agent bullets run on Modal, so they need what python bullets need."""
    return modal_exec.enabled()


def _split_env() -> tuple[list[str], list[str], list[str]]:
    """The deployment's apt and pip extras, and anything refused among them."""
    apt = [a for a in re.split(r"[,\s]+", os.environ.get(APT_ENV, "").strip()) if a]
    pip, bad_pip = modal_exec.split_requirements(os.environ.get(PIP_ENV, ""))
    bad = [a for a in apt if not _APT_NAME.match(a)] + bad_pip
    return [a for a in apt if _APT_NAME.match(a)], pip, bad


def secrets() -> list[str]:
    return [s for s in re.split(r"[,\s]+", os.environ.get(SECRETS_ENV, "").strip()) if s]


def rejected() -> list[str]:
    """Entries in the deployment's extras that are not package names."""
    return _split_env()[2]


_app = None
_image = None
_browser_image = None


def _lookup(browser: bool = False):
    """The Modal app and agent image, created once and reused. With *browser*,
    the image that also has a display, Chromium and noVNC.

    Deferred rather than done at import, like `modal_exec._lookup`: both hit
    the network, and a server with no Modal credentials must still start.
    """
    global _app, _image, _browser_image
    if _app is None:
        import modal

        apt, pip, _ = _split_env()
        _app = modal.App.lookup(APP_NAME, create_if_missing=True)
        _image = (
            modal.Image.debian_slim(python_version="3.12")
            # git and curl because an agent reaches for them.
            .apt_install("git", "curl", "ca-certificates", *apt)
            # Node for `npx` servers — 22 from NodeSource, since Debian's 18 is
            # too old for many of them (Playwright's needs 20+).
            .run_commands(
                "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
                "apt-get install -y nodejs",
                "node --version",
            )
            # A single prebuilt binary; npm only picks the one for this platform.
            .run_commands(f"npm install -g opencode-ai@{OPENCODE_VERSION}", "opencode --version")
            # uv for `uvx` servers.
            .uv_pip_install("uv", *pip)
            .add_local_file(AGENT_DIR / "run_agent.py", "/opt/agent/run_agent.py", copy=True)
        )
    if browser:
        if _browser_image is None:
            # Its own image, so agents without a browser do not pull Chromium.
            _browser_image = _image.apt_install(
                "xvfb", "x11vnc", "novnc", "websockify", "chromium", "fonts-liberation"
            )
        return _app, _browser_image
    return _app, _image


def _secret_env(
    provider: str, api_key: str | None, steps: int = STEP_LIMIT, model: str | None = None
) -> dict[str, str]:
    """What the sandbox needs to call the model: its name and the key.

    A *model* the person chose wins over the deployment's `AGENT_MODEL`, so
    an agent bullet answers with the same model as the prompt bullets around it.

    The key goes in under the name OpenCode reads for that provider, which is
    not always this server's: Google's is `GOOGLE_GENERATIVE_AI_API_KEY`, and
    `GEMINI_API_KEY` alone fails with "API key is missing".
    """
    key = api_key or ""
    if not key:
        raise RuntimeError(f"No API key for '{provider}'. Enter one in settings.")
    if model:
        model = f"{_OPENCODE_PROVIDER[provider]}/{model}"
    else:
        model = os.environ.get("AGENT_MODEL") or f"{_OPENCODE_PROVIDER[provider]}/{model_name(provider)}"
    return {
        _OPENCODE_KEY_ENV.get(provider, ENV_KEYS[provider]): key,
        "AGENT_MODEL": model,
        "AGENT_STEP_LIMIT": str(steps),
        "AGENT_MCP_START_SECONDS": str(MCP_START_SECONDS),
    }


class _Log:
    """The run's events, each at its offset in seconds from the start."""

    def __init__(self, on_line: Callable[[str], None] | None = None) -> None:
        self.t0 = time.time()
        self.events: list[tuple[float, str, str]] = []
        self.on_line = on_line

    def add(self, source: str, message: str, at: float | None = None) -> None:
        self.events.append(((at if at is not None else time.time()) - self.t0, source, message))
        if at is None:
            self.live(source, message)

    def live(self, source: str, message: str) -> None:
        """Pass one line on as it happens. Stamped on arrival: the final log
        re-places sandbox events on the sandbox's own clock, but live, now is
        close enough and all there is."""
        if self.on_line:
            try:
                self.on_line(f"[{time.time() - self.t0:7.1f}s] {source}: {message}")
            except Exception:  # noqa: BLE001 — a log line is never worth failing the run
                pass

    def lines(self) -> list[str]:
        # Stable, so events stamped the same instant keep the order they came in.
        ordered = sorted(self.events, key=lambda e: e[0])
        out = [f"[{t:7.1f}s] {source}: {message}" for t, source, message in ordered]
        if len(out) > MAX_LOG_LINES:
            keep = MAX_LOG_LINES // 2
            out = out[:keep] + [f"… {len(out) - 2 * keep} lines elided …"] + out[-keep:]
        return out

    def tail(self, n: int = 40) -> str:
        return "\n".join(self.lines()[-n:])


class AgentError(RuntimeError):
    """A failed agent run, with the log up to where it stopped.

    A failure is when the log matters most, and pbt reports a failed bullet by
    its error message alone — so the tail of the log goes into the message.
    """

    def __init__(self, message: str, log: _Log) -> None:
        super().__init__(f"{message}\n\nLog (last lines):\n{log.tail()}")


def _read(sb, path: str) -> str:
    """A file in the sandbox, or "" if it is not there."""
    p = sb.exec("cat", path)
    text = p.stdout.read()
    return text if p.wait() == 0 else ""


def _start_browser(sb, log: _Log) -> str:
    """Start the display, Chromium and noVNC in *sb*; the URL to watch it at.

    The VNC password rides in the URL, so the link alone lets you in and the
    tunnel alone does not. VNC passwords stop at eight characters.
    """
    password = secrets_mod.token_hex(4)
    script = f"""
set -e
Xvfb {DISPLAY} -screen 0 {SCREEN} >/tmp/xvfb.log 2>&1 &
for i in $(seq 50); do [ -e /tmp/.X11-unix/X{DISPLAY[1:]} ] && break; sleep 0.1; done
DISPLAY={DISPLAY} chromium --no-sandbox --disable-dev-shm-usage --no-first-run \
  --no-default-browser-check --user-data-dir=/root/.browser \
  --window-position=0,0 --window-size={SCREEN.rsplit("x", 1)[0].replace("x", ",")} \
  --remote-debugging-port={CDP_PORT} about:blank >/tmp/chromium.log 2>&1 &
x11vnc -storepasswd {password} /tmp/vncpass >/dev/null 2>&1
x11vnc -display {DISPLAY} -forever -shared -rfbauth /tmp/vncpass -rfbport 5900 >/tmp/x11vnc.log 2>&1 &
websockify --web /usr/share/novnc {VNC_PORT} localhost:5900 >/tmp/websockify.log 2>&1 &
for i in $(seq 150); do curl -sf http://localhost:{CDP_PORT}/json/version >/dev/null && exit 0; sleep 0.2; done
echo "Chromium did not open its DevTools port:"; tail -n 20 /tmp/chromium.log; exit 1
"""
    log.add("browser", "starting display, Chromium and noVNC")
    p = sb.exec("bash", "-c", script)
    out = p.stdout.read() + p.stderr.read()
    if p.wait() != 0:
        log.add("browser", f"failed to start:\n{out.strip()[-2000:]}")
        raise AgentError("The remote browser did not start.", log)
    url = (
        f"{sb.tunnels()[VNC_PORT].url}/vnc.html"
        f"?autoconnect=1&resize=scale&password={password}"
    )
    log.add("browser", f"watch live at {url}")
    return url


def _run_sandbox(
    task: str,
    mcp_server: str,
    env: dict[str, str],
    json_answer: bool,
    on_line: Callable[[str], None] | None = None,
    want_files: bool = False,
    browser: bool = False,
) -> dict:
    """Run the agent on *task* in a fresh sandbox. Blocking — call it off the loop."""
    import modal

    log = _Log(on_line)
    files: list[pbt.File] = []
    app, image = _lookup(browser)
    command = shlex.split(mcp_server) if mcp_server.strip() else []

    log.add("modal", f"creating sandbox (app {APP_NAME}, {CPU} cpu, {MEMORY_MB} MB, timeout {TIMEOUT_SECONDS}s)")
    sb = modal.Sandbox.create(
        "sleep",
        "infinity",
        app=app,
        image=image,
        timeout=TIMEOUT_SECONDS,
        workdir="/root",
        cpu=CPU,
        memory=MEMORY_MB,
        secrets=[modal.Secret.from_dict(env), *(modal.Secret.from_name(s) for s in secrets())],
        encrypted_ports=[VNC_PORT] if browser else [],
    )
    log.add("modal", f"sandbox {sb.object_id} up")
    result: dict[str, Any] = {}
    try:
        if browser:
            _start_browser(sb, log)
        # Through stdin, not argv: a rendered bullet can be long, and nothing
        # in it should be read by a shell.
        put = sb.exec("sh", "-c", f"cat > {TASK_PATH}")
        put.stdin.write(task.encode())
        put.stdin.write_eof()
        put.stdin.drain()
        put.wait()

        if command:
            log.add("modal", f"starting MCP server: {shlex.join(command)}")
        backoff = f", {env['AGENT_BACKOFF_SECONDS']}s between model calls" if env.get("AGENT_BACKOFF_SECONDS") else ""
        log.add("agent", f"starting ({env['AGENT_MODEL']}, up to {env['AGENT_STEP_LIMIT']} steps{backoff}), task of {len(task)} chars")
        launched = time.time()
        # The server command goes as arguments, so no shell reads it. Read line
        # by line as it runs: each flagged line is a log event, relayed live.
        p = sb.exec("python", "/opt/agent/run_agent.py", TASK_PATH, RESULT_PATH, *command, bufsize=1)
        printed: list[str] = []
        for line in p.stdout:
            if line.startswith(LIVE_MARKER):
                try:
                    event = json.loads(line[len(LIVE_MARKER):])
                    log.live(event.get("source", "agent"), event.get("message", ""))
                except ValueError:
                    pass
            else:
                printed.append(line)
        out = "".join(printed)
        err = p.stderr.read()
        code = p.wait()

        raw = _read(sb, RESULT_PATH)
        if not raw.strip():
            log.add("agent", f"exited {code} without a result:\n{(err or out or '').strip()[-3000:]}")
            raise AgentError(f"The agent exited {code} without a result.", log)
        result = json.loads(raw)
        # The sandbox's clock is not this one. Its events are placed by lining
        # up the agent's own start with the moment it was launched from here.
        offset = launched - float(result.get("started", launched))
        for e in result.get("events", []):
            log.add(e.get("source", "agent"), e.get("message", ""), float(e["ts"]) + offset)
        log.add(
            "agent",
            f"finished: {result.get('exit_status') or 'unknown'} after {result.get('steps', '?')} steps"
            + (f", ${result['cost']:.4f}" if result.get("cost") else ""),
        )
        # Before teardown, while the sandbox still has them — and only once
        # there is an answer, since a failed run fails whatever it saved.
        answer = _answer(result, json_answer, log)
        if want_files:
            files = _collect_files(sb, log)
    finally:
        sb.terminate()
        log.add("modal", "sandbox terminated")

    out: dict[str, Any] = {"output": answer}
    if want_files:
        out["files"] = files
    return {
        **out,
        "logs": log.lines(),
        "run_time": round(time.time() - log.t0, 1),
    }


def _collect_files(sb, log: _Log) -> list[pbt.File]:
    """The files the agent saved in OUTPUTS_DIR, as pbt files.

    Subfolders are flattened into the name (`plots/a.png` → `plots_a.png`):
    pbt file names never carry a path. A bullet that promised files and saved
    none fails, because everything downstream is waiting to be handed them.
    """
    p = sb.exec("find", OUTPUTS_DIR, "-type", "f", "-printf", "%s\t%P\n")
    listing = p.stdout.read()
    p.wait()
    found = []
    for line in listing.splitlines():
        size, _, rel = line.partition("\t")
        if rel and size.isdigit():
            found.append((rel, int(size)))
    found.sort()
    if not found:
        raise AgentError(
            f"This bullet is set to produce files, but the agent saved none in {OUTPUTS_DIR}/.", log
        )

    files: list[pbt.File] = []
    taken: set[str] = set()
    total = 0
    for rel, size in found:
        if len(files) >= MAX_FILES or total + size > MAX_FILES_BYTES:
            log.add("files", f"skipped {rel} ({size} bytes): over the limit of {MAX_FILES} files / {MAX_FILES_BYTES} bytes")
            continue
        data = sb.filesystem.read_bytes(f"{OUTPUTS_DIR}/{rel}")
        name = rel.replace("/", "_")
        stem, dot, ext = name.rpartition(".")
        n = 2
        while name in taken:
            name = f"{stem}_{n}{dot}{ext}" if dot else f"{ext}_{n}"
            n += 1
        taken.add(name)
        total += len(data)
        files.append(pbt.File(data, name=name))
        log.add("files", f"collected {name} ({len(data)} bytes)")
    return files


def _answer(result: dict[str, Any], json_answer: bool, log: _Log) -> Any:
    """The submission — parsed, if the bullet enforces JSON — or an error saying
    why there is none."""
    submission = (result.get("submission") or "").strip()
    if result.get("exit_status") == "MCPServerFailed":
        raise AgentError(f"The MCP server did not start: {result.get('error', '')}", log)
    if result.get("exit_status") == "Answered" and submission:
        if not json_answer:
            return submission[:MAX_OUTPUT_CHARS]
        try:
            return pbt_json(submission)
        except ValueError as exc:
            raise AgentError(f"The agent's answer is not valid JSON: {exc}", log) from exc
    status = result.get("exit_status") or "an unknown reason"
    said = (result.get("error") or result.get("last_message") or "").strip()[-2000:]
    if status.startswith("StepLimit"):
        headline = f"The agent used all {status.split()[-1]} of its steps without answering, so it was stopped."
    else:
        headline = f"The agent stopped without an answer ({status}, after {result.get('steps', '?')} steps)."
    raise AgentError(
        headline
        + (f" Last it said:\n{said}" if said else ""),
        log,
    )


async def _run(
    task: str,
    mcp_server: str,
    env: dict[str, str],
    json_answer: bool,
    on_line: Callable[[str], None] | None = None,
    want_files: bool = False,
    browser: bool = False,
) -> dict:
    if not enabled():
        raise RuntimeError(
            "Agent bullets run on Modal, which is not configured on this "
            "server. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET."
        )
    if refused := rejected():
        raise RuntimeError(
            f"{APT_ENV} / {PIP_ENV} list something that is not a package name: "
            f"{', '.join(refused)}. Nothing is run rather than running without it."
        )
    if not task.strip():
        raise RuntimeError("An agent bullet needs a task — its text, or something from below it.")
    # Modal's client is blocking, and pbt runs independent branches
    # concurrently — keep one agent from stalling the others.
    # Returned as it is: pbt's cache stores a dict — and the files in it, by
    # hash — and hands the same back on a hit.
    return await asyncio.to_thread(
        _run_sandbox, task, mcp_server, env, json_answer, on_line, want_files, browser
    )


# Registered on import, like `modal_exec`: importing this module is what teaches
# pbt the kind. The rendered text is a task in natural language, so the run's
# global instruction is welcome here.
@pbt.model_kind(MODEL_TYPE, config_keys={MCP_KEY, STEPS_KEY, FILES_KEY, BACKOFF_KEY, BROWSER_KEY})
async def execute(rendered: str, call: pbt.ModelCall) -> dict:
    """Hand the rendered bullet to a coding agent, with its MCP server if it names one.

    Returns ``{"output", "logs", "run_time"}`` — see the module docstring. A
    structured value, so pbt passes it on as it is rather than parsing it.

    The key is resolved *outside* `call.compute`, so a missing one fails the
    bullet before anything is cached or started, and never enters the cache key.
    """
    mcp_server = str(call.spec.config.get(MCP_KEY, ""))
    run = _settings.get()
    # The bullet's own limit, if it set one, over the run's.
    try:
        own = int(call.spec.config.get(STEPS_KEY) or 0)
    except (TypeError, ValueError):
        own = 0
    env = _secret_env(
        run.provider, run.api_key, steps_for(own) if own > 0 else run.steps, run.model
    )
    name = call.spec.name
    want_files = bool(call.spec.config.get(FILES_KEY))
    if want_files:
        env["AGENT_OUTPUT_FILES"] = "1"
    browser = bool(call.spec.config.get(BROWSER_KEY))
    if browser:
        # Every process in the sandbox — the agent's shell, the MCP server —
        # draws on the display the person is watching.
        env.update(AGENT_REMOTE_BROWSER="1", DISPLAY=DISPLAY, AGENT_CDP_URL=f"http://localhost:{CDP_PORT}")
    # Pacing only — though, being in `config()`, changing it still misses the cache.
    try:
        backoff = min(float(call.spec.config.get(BACKOFF_KEY) or 0), MAX_BACKOFF)
    except (TypeError, ValueError):
        backoff = 0
    if backoff > 0:
        env["AGENT_BACKOFF_SECONDS"] = "%g" % backoff
    on_line = (lambda line: run.on_log(name, line)) if run.on_log else None

    json_answer = call.spec.output_format == "json"
    # Keyed on `rendered` exactly: pbt stores the result under the key it builds
    # from the rendered prompt, so any other text here is looked up but never
    # found. The toolbox, steps, files and chosen model are all in the bullet's
    # `config()`, which pbt already folds into that key.
    raw = await call.compute(
        rendered,
        compute=lambda: _run(
            rendered.strip(), mcp_server, env, json_answer, on_line, want_files, browser
        ),
    )
    # An entry cached before results were returned as dicts is JSON text.
    return raw if isinstance(raw, dict) else json.loads(raw)
