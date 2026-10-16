"""
An agent bullet whose disk outlives the run: a Hermes agent on one Agent37
instance shared by every such bullet.

    pbt user  → Hermes profile  ``u-<sha256(user)[:20]>``  (own config, memory, sessions)
    node      → session_id      ``sha256(node id)[:32]``   (own chat, own folder)

The user is the signed-in email, else the browser's session id. A profile is
created on first use (``hermes profile create --clone``, memories wiped, its
terminal homed in the profile) and holds that user's model key, so one user's
runs never use another's. Each node keeps its own conversation and works in
``~/.hermes/profiles/<profile>/pbt/<session id>/``.

    Agent37 instance    (template `hermes-vnc-desktop`, one, shared, auto-sleep)
    ├── Hermes          answers POST /v1/responses with {profile, session_id}
    ├── Chromium        on a visible desktop the agent's browser tool drives
    └── noVNC :6901     that desktop, watchable from a signed URL

Lifecycle: the turn is sent straight to the instance's URL — a sleeping
instance wakes on the request itself. Only when that send cannot reach it (no
instance yet, stopped, deleted, gateway down) is the instance created or
started, waited on until healthy, and the send made again. The instance is
created with ``auto_sleep`` so it drops to disk-only billing after
``AGENT37_IDLE_SECONDS`` of no traffic; nothing here stops it.

Profiles separate config, memory and history, not the disk: they are one
Linux user, so an agent can still read another profile's files by path. A user
who sets their own Agent37 key in Settings gets an instance of their own.
Delete the instance on Agent37 to start everyone over.

It is emitted as ``{{ config(model_type="agent37", session="<bullet id>") }}``.
MCP servers, the Modal browser and back-off are the Modal agent's and are not
offered here; the step limit and ``produces_files=True`` are. Files are saved in
the bullet's ``outputs/`` folder, emptied at the start of each run so only that
run's files are handed on.

The run's provider, key, steps and log sink come from `agent_exec`'s run
context, which `main.run` binds for every agent bullet alike.

What Agent37 offers, which calls this uses, and the errors it branches on:
server/agent37.md.

The template is built once per Agent37 workspace (see server/README.md):

    git clone https://github.com/agent37-platform/examples
    cd examples/custom-images/hermes-vnc-desktop
    npx agent37 templates build . --name hermes-vnc-desktop --default-port 3737
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import shlex
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import httpx
import pbt
from pbt.executor.run_context import parse_json_output as pbt_json

from . import agent_exec
from .models import model_name

MODEL_TYPE = "agent37"

# The config key naming the bullet's session on the shared instance. The bullet's id.
SESSION_KEY = "session"
# The config key holding the bullet's own step limit. Absent means the run's.
STEPS_KEY = agent_exec.STEPS_KEY
# The config key saying the bullet must hand on files, not only text.
FILES_KEY = agent_exec.FILES_KEY
# The config key asking for the machine's desktop to be linked in the log.
BROWSER_KEY = agent_exec.BROWSER_KEY
# The config key asking for a public URL to a port of the bullet's own.
PUBLIC_KEY = "public_url"
# A bullet's public port is picked from this range by its session id, so it is
# the same every run; Agent37 allows 50 public ports per instance.
PUBLIC_PORT_BASE = 20000
PUBLIC_PORT_SPAN = 10000

# Each node's folder inside its user's Hermes profile, then the session id. Its
# `outputs/` is where a node asked for files saves them, emptied at the start of
# every run so only that run's files are handed on.
WORK_DIR = "pbt"
MAX_FILES = agent_exec.MAX_FILES
MAX_FILES_BYTES = agent_exec.MAX_FILES_BYTES

API = os.environ.get("AGENT37_API_URL", "https://api.agent37.com/v1")
API_KEY_ENV = "AGENT37_API_KEY"
TEMPLATE = os.environ.get("AGENT37_TEMPLATE", "hermes-vnc-desktop")
# A cap on what one instance may spend on Agent37's managed services (web
# search, and the model when AGENT37_MANAGED_MODEL is set). Compute is billed
# apart from it, by the minute, while the instance runs.
BUDGET_MICROS = int(os.environ.get("AGENT37_BUDGET_MICROS", "1000000"))
# Run on Agent37's managed model, billed to the workspace, instead of the run's
# own provider and key.
MANAGED_MODEL = os.environ.get("AGENT37_MANAGED_MODEL", "").lower() in ("1", "true", "yes")
# The shared instance's name, found again by it on every server start.
INSTANCE_NAME = os.environ.get("AGENT37_INSTANCE_NAME", "pbt-shared")
# An existing instance to share instead, by id; wins over the name for the
# server's own key. A user's own key still finds theirs by name.
INSTANCE_ID = os.environ.get("AGENT37_INSTANCE_ID", "9c7c78hguk")
# How long the instance may see no traffic before Agent37 puts it to sleep
# (disk-only billing). Agent37 allows 300 to 86400.
IDLE_SECONDS = int(os.environ.get("AGENT37_IDLE_SECONDS", "600"))

# The longest a turn may run before it is cancelled and the bullet fails.
TIMEOUT_SECONDS = agent_exec.TIMEOUT_SECONDS
# How long a bullet waits for a free instance when the workspace is at its
# instance limit.
QUEUE_SECONDS = int(os.environ.get("AGENT37_QUEUE_SECONDS", "900"))
# How long an instance's agent may take to report healthy once it is running.
READY_SECONDS = int(os.environ.get("AGENT_READY_SECONDS", "240"))
# How long a stopped instance may take to be running again. Usually seconds,
# but Agent37 moves it to another host when its own is full, which takes
# minutes per 10 GB of disk.
START_SECONDS = int(os.environ.get("AGENT37_START_SECONDS", "900"))

# The noVNC port of the desktop template, and how long its link opens it for.
# The token in it grants full control and cannot be revoked, so it is short.
DESKTOP_PORT = 6901
DESKTOP_TTL_SECONDS = int(os.environ.get("AGENT37_DESKTOP_TTL_SECONDS", "300"))

MAX_OUTPUT_CHARS = agent_exec.MAX_OUTPUT_CHARS
MAX_LOG_LINES = agent_exec.MAX_LOG_LINES
# How much of a tool call's label goes into a log line.
LABEL_CHARS = 300

# Each provider's OpenAI-compatible endpoint, which Hermes is pointed at as a
# custom provider (see `_prepare`).
_PROVIDER_URL = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}
# The custom provider's name in Hermes' config.
PROVIDER_NAME = "Run"
# The key variable Hermes' built-in provider reads, for a provider it handles
# itself whatever the config says: an `openrouter.ai` URL goes through its own
# OpenRouter client, which ignores the custom provider's key and fails `401
# User not found` without this.
_HERMES_KEY_ENV = {"openrouter": "OPENROUTER_API_KEY"}

FINISHING = """

---

## How to finish

You work on a Linux machine with a terminal, files and a web browser. The
person can see your browser's screen: if a site needs them (a sign-in, a code,
a CAPTCHA), say so and stop rather than asking for a password in chat.

This conversation and your folder {workdir}/ are kept across runs: what you
leave there is there the next time you are given a task, so check it for
earlier work. Work in that folder; other people's agents share this machine —
leave everything outside it alone.

You have about {steps} steps; leave the last of them for the answer rather
than running out mid-task.
Your final message is what the person reads, so make it the answer itself —
not a description of what you did or where you put it.
"""

PUBLIC_NOTE = """
Anything you serve on port {port} (bind 0.0.0.0) is public at {url} — no
sign-in, for anyone with the link, and the same link every run. Use it when the
task wants something reachable (a site, an API, a webhook); keep the server
running in the background (e.g. `nohup … &`) so it outlives this turn, and give
the link in your answer.
"""

FILES_NOTE = """
This task must produce files. Save every file the person should get — images,
documents, data — in {outputs}/ (it exists already, and is emptied before each
run; keep anything you want for later elsewhere). They are handed on as files,
so save at least one; your final message is handed on as text beside them, so
say what each file is rather than pasting its contents.
"""


def config_line(
    session: str,
    steps: int = 0,
    produces_files: bool = False,
    browser: bool = False,
    public_url: bool = False,
) -> str:
    """The bullet's config line, naming its session and, when set, its step
    limit, whether it hands on files, links its desktop, or gets a public URL."""
    args = ['model_type="%s"' % MODEL_TYPE, "%s=%s" % (SESSION_KEY, json.dumps(session))]
    if steps and steps > 0:
        args.append("%s=%d" % (STEPS_KEY, int(steps)))
    if produces_files:
        args.append("%s=True" % FILES_KEY)
    if browser:
        args.append("%s=True" % BROWSER_KEY)
    if public_url:
        args.append("%s=True" % PUBLIC_KEY)
    return "{{ config(%s) }}" % ", ".join(args)


def enabled() -> bool:
    """Whether the server has its own Agent37 key, for the shared instance. A
    user can still run on their own workspace by setting their key in Settings."""
    return bool(os.environ.get(API_KEY_ENV))


@dataclass(frozen=True)
class _Provider:
    """The run's model, as Hermes' custom provider: endpoint, key, model id."""

    base_url: str
    api_key: str
    model: str
    # Also written to ~/.hermes/.env under this name — see `_HERMES_KEY_ENV`.
    key_env: str | None = None


def _provider(provider: str, api_key: str | None, model: str | None) -> _Provider | None:
    """The run's provider and key for Hermes, or None for Agent37's managed model."""
    if MANAGED_MODEL:
        return None
    if not api_key:
        raise RuntimeError(f"No API key for '{provider}'. Enter one in settings.")
    if provider not in _PROVIDER_URL:
        raise RuntimeError(f"Agent bullets cannot run on '{provider}'.")
    return _Provider(
        _PROVIDER_URL[provider], api_key, model or model_name(provider), _HERMES_KEY_ENV.get(provider)
    )


class _Log:
    """The run's events, each at its offset in seconds from the start."""

    def __init__(self, on_line: Callable[[str], None] | None = None) -> None:
        self.t0 = time.time()
        self.lines_: list[str] = []
        self.on_line = on_line

    def add(self, source: str, message: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {source}: {message}"
        self.lines_.append(line)
        if self.on_line:
            try:
                self.on_line(line)
            except Exception:  # noqa: BLE001 — a log line is never worth failing the run
                pass

    def lines(self) -> list[str]:
        out = self.lines_
        if len(out) > MAX_LOG_LINES:
            keep = MAX_LOG_LINES // 2
            out = out[:keep] + [f"… {len(out) - 2 * keep} lines elided …"] + out[-keep:]
        return out

    def tail(self, n: int = 40) -> str:
        return "\n".join(self.lines()[-n:])


class AgentError(RuntimeError):
    """A failed agent run, with the log up to where it stopped."""

    def __init__(self, message: str, log: _Log) -> None:
        super().__init__(f"{message}\n\nLog (last lines):\n{log.tail()}")


class Agent37Error(RuntimeError):
    """An Agent37 API error, with its status and error body kept for retries."""

    def __init__(self, message: str, status: int, err: dict) -> None:
        super().__init__(message)
        self.status = status
        self.code = err.get("code") or ""
        self.err = err


def _raise_for(r: httpx.Response) -> None:
    if r.status_code < 400:
        return
    err: dict = {}
    try:
        err = r.json().get("error") or {}
        detail = f"{err.get('code', '')}: {err.get('message', '')}".strip(": ")
    except (ValueError, AttributeError):
        detail = r.text[:500]
    raise Agent37Error(
        f"Agent37 {r.request.method} {r.request.url.path} → {r.status_code} {detail}", r.status_code, err
    )


def _unreachable(exc: BaseException) -> bool:
    """Whether *exc* means the instance is not there to take the turn — none,
    stopped, deleted, or its gateway down — so it should be brought up and the
    turn sent again. A busy session or a bad request is not."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, Agent37Error):
        if exc.code in ("session_busy", "profile_not_found"):
            return False
        return exc.status in (404, 409, 410, 502, 503, 504)
    return False


class _Instance:
    """One Agent37 instance: the Sandbox API by id, the Agent API by its URL."""

    def __init__(self, client: httpx.Client, key: str, data: dict) -> None:
        self.client = client
        self.key = key
        self.id = data["id"]
        self.url = data["url"].rstrip("/")

    def _api(self, method: str, path: str, **kw) -> dict:
        r = self.client.request(
            method, f"{API}/instances/{self.id}{path}",
            headers={"Authorization": f"Bearer {self.key}"}, **kw,
        )
        _raise_for(r)
        return r.json()

    def _agent(self, method: str, path: str, **kw) -> httpx.Response:
        r = self.client.request(method, f"{self.url}{path}", headers={"X-Agent37-Key": self.key}, **kw)
        _raise_for(r)
        return r

    def status(self) -> str:
        return self._api("GET", "").get("status", "")

    def exec(self, command: str) -> dict:
        return self._api("POST", "/exec", json={"command": command}, timeout=800)

    def start(self, log: _Log) -> None:
        """Bring a stopped or sleeping instance up, retrying while its sleep
        checkpoint is still being written, and wait until it reads running."""
        log.add("agent37", f"starting instance {self.id}")
        t0 = time.time()
        deadline = t0 + START_SECONDS
        while True:
            r = self.client.post(
                f"{API}/instances/{self.id}/start",
                headers={"Authorization": f"Bearer {self.key}"}, timeout=600,
            )
            if r.status_code == 409 and "try_again" in r.text and time.time() < deadline:
                time.sleep(5)
                continue
            _raise_for(r)
            break
        said = 0.0
        while (s := self.status()) != "running":
            if time.time() > deadline:
                raise AgentError(f"instance {self.id} was still '{s}' after {START_SECONDS}s", log)
            # A line every 30s, so a slow start does not look like a hang.
            if time.time() - said >= 30:
                log.add("agent37", f"instance {self.id} is '{s}' ({time.time() - t0:.0f}s so far)")
                said = time.time()
            time.sleep(3)
        log.add("agent37", f"instance {self.id} running after {time.time() - t0:.0f}s")

    def wait_healthy(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while True:
            try:
                r = self.client.get(f"{self.url}/v1/health", headers={"X-Agent37-Key": self.key}, timeout=10)
                if r.status_code == 200 and r.json().get("healthy") is True:
                    return
            except (httpx.HTTPError, ValueError):
                pass
            if time.time() > deadline:
                raise RuntimeError(f"the agent was not healthy after {seconds:.0f}s")
            time.sleep(2)


def session_id(key: str) -> str:
    """The bullet's Hermes session id: 32 hex characters, as Agent37 issues them."""
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class _Workspace:
    """What is known about one Agent37 workspace's shared instance — the
    server's, or a user's own when they set their key in Settings."""

    def __init__(self) -> None:
        # The instance's `{id, url}`, once found or created, so later runs send
        # straight to it without listing the workspace. Cleared when it proves gone.
        self.shared: dict | None = None
        # Held while the instance is found, created or started, so concurrent
        # bullets bring it up once rather than each creating their own.
        self.up_lock = threading.Lock()
        # Held while Hermes' config is written; `config set` is the instance's,
        # not the session's, so two bullets must not write it at once.
        self.config_lock = threading.Lock()
        # The config last written — provider, and the step cap — so it is only
        # rewritten when it changes. Per profile.
        self.configured: dict[str, dict[str, Any]] = {}
        # The profiles known to exist on the instance.
        self.profiles: set[str] = set()


# One per Agent37 key seen, keyed by the key's hash rather than the key.
_workspaces: dict[str, _Workspace] = {}
_workspaces_lock = threading.Lock()


def _workspace(key: str) -> _Workspace:
    h = hashlib.sha256(key.encode()).hexdigest()
    with _workspaces_lock:
        return _workspaces.setdefault(h, _Workspace())


def _find(client: httpx.Client, key: str) -> dict | None:
    """The workspace's shared instance, if there is one."""
    r = client.get(f"{API}/instances", headers={"Authorization": f"Bearer {key}"}, timeout=60)
    _raise_for(r)
    pinned = INSTANCE_ID if key == os.environ.get(API_KEY_ENV) else ""
    for inst in r.json().get("data", []):
        if (inst.get("id") == pinned if pinned else inst.get("name") == INSTANCE_NAME) and inst.get("status") not in ("deleted", "deleting"):
            return inst
    return None


def _create(client: httpx.Client, key: str, log: _Log) -> dict:
    """Create the shared instance with auto-sleep on, waiting for a free slot
    while the workspace is at its instance limit."""
    deadline = time.time() + QUEUE_SECONDS
    waiting = False
    while True:
        r = client.post(
            f"{API}/instances",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "template": TEMPLATE,
                "name": INSTANCE_NAME,
                "budget": {"credit_micros": BUDGET_MICROS},
                "auto_sleep": True,
                "idle_timeout_seconds": IDLE_SECONDS,
            },
            timeout=600,
        )
        limited = r.status_code == 409 and "instance_limit" in r.text
        if not limited or time.time() > deadline:
            _raise_for(r)
            return r.json()
        if not waiting:
            log.add("agent37", "at the workspace's instance limit; waiting for a free slot")
            waiting = True
        time.sleep(10)


def _ensure_auto_sleep(client: httpx.Client, key: str, found: dict, log: _Log) -> None:
    """Turn auto-sleep on for an instance made before it was, or by hand."""
    if found.get("auto_sleep") is True and found.get("idle_timeout_seconds") == IDLE_SECONDS:
        return
    r = client.patch(
        f"{API}/instances/{found['id']}",
        headers={"Authorization": f"Bearer {key}"},
        json={"auto_sleep": True, "idle_timeout_seconds": IDLE_SECONDS},
        timeout=60,
    )
    _raise_for(r)
    log.add("agent37", f"auto-sleep on after {IDLE_SECONDS}s idle")


def _bring_up(client: httpx.Client, key: str, ws: _Workspace, log: _Log) -> _Instance:
    """Find, create or start the shared instance and wait until its agent is healthy."""
    with ws.up_lock:
        found = _find(client, key)
        if found:
            inst = _Instance(client, key, found)
            status = found.get("status")
            log.add("agent37", f"instance {inst.id} is '{status}'")
            _ensure_auto_sleep(client, key, found, log)
            # A sleeper wakes on the health check below; stopped stays down until started.
            if status not in ("running", "sleeping"):
                inst.start(log)
        else:
            log.add("agent37", f"no shared instance yet; creating {INSTANCE_NAME} (template {TEMPLATE})")
            inst = _Instance(client, key, _create(client, key, log))
            ws.configured.clear()
            ws.profiles.clear()
        log.add("agent37", f"instance {inst.id} up; waiting for the agent")
        inst.wait_healthy(READY_SECONDS)
        ws.shared = {"id": inst.id, "url": inst.url}
        return inst


def _public_url(inst: _Instance, session: str) -> tuple[int, str]:
    """The bullet's public port and its permanent URL, made on first use.

    Found again by label, so it is the same link every run. The port starts at
    one picked by the session id and moves past ports another bullet holds.
    """
    label = f"pbt-{session[:24]}"
    held = inst._api("GET", "/public-ports")
    entries = held.get("data", held) if isinstance(held, dict) else held
    for e in entries or []:
        if e.get("label") == label:
            return int(e["port"]), e["url"]
    taken = {int(e.get("port", 0)) for e in entries or []}
    port = PUBLIC_PORT_BASE + int(session[:8], 16) % PUBLIC_PORT_SPAN
    while port in taken:
        port = PUBLIC_PORT_BASE + (port - PUBLIC_PORT_BASE + 1) % PUBLIC_PORT_SPAN
    made = inst._api("POST", "/public-ports", json={"port": port, "label": label})
    return port, made["url"]


def _desktop_url(inst: _Instance) -> str:
    """A signed link to the instance's desktop, at noVNC's page rather than `/`."""
    signed = inst._api("POST", "/signed-url", json={"port": DESKTOP_PORT, "ttl_seconds": DESKTOP_TTL_SECONDS})
    u = urlparse(signed["url"])
    token = parse_qs(u.query)["a37_token"][0]
    return f"https://{u.netloc}/vnc.html?a37_token={token}&autoconnect=1&resize=scale"


def profile_for(user: str) -> str:
    """The pbt user's Hermes profile: their own config, memory and sessions on
    the shared instance. A hash, so no email ends up in a path."""
    return "u-" + hashlib.sha256(user.encode()).hexdigest()[:20]


def _prepare(
    inst: _Instance,
    ws: _Workspace,
    profile: str,
    session: str,
    steps: int,
    provider: _Provider | None,
    want_files: bool,
    log: _Log,
) -> str:
    """Create the user's profile if this server has not seen it yet, make the
    node's folder inside it, empty its outputs, and write the profile's Hermes
    config when the run's provider or steps need it. Returns the folder's
    absolute path.

    Config is the profile's, shared by that user's nodes, so it is written
    under a lock and only when it changes. The step cap only ever rises — a
    node's own limit is in its task text — so one node never cuts short
    another that is mid-turn.
    """
    pdir = f"~/.hermes/profiles/{profile}"
    workdir = f"{pdir}/{WORK_DIR}/{session}"
    hermes = f"hermes --profile={profile}"
    parts = []
    with ws.config_lock:
        if profile not in ws.profiles:
            # Idempotent: a profile made by an earlier server process is kept.
            parts.append(
                f"(test -d {pdir} || (hermes profile create {profile} --clone"
                f" && rm -f {pdir}/memories/*.md"
                f" && {hermes} config set terminal.home_mode profile))"
            )
        done = ws.configured.setdefault(profile, {})
        cap = max(steps, done.get("steps", 0))
        sets = []
        if cap != done.get("steps"):
            sets.append(("agent.max_turns", str(cap)))
        new_provider = provider is not None and provider != done.get("provider")
        if new_provider:
            sets += [
                ("custom_providers", json.dumps([{
                    "name": PROVIDER_NAME,
                    "base_url": provider.base_url,
                    "api_key": provider.api_key,
                    "api_mode": "chat_completions",
                    "model": provider.model,
                }])),
                ("model.provider", f"custom:{PROVIDER_NAME}"),
                ("model.default", provider.model),
            ]
        parts += [f"{hermes} config set {k} {shlex.quote(v)}" for k, v in sets]
        if new_provider and provider.key_env:
            # Replaced rather than appended, so a kept disk does not pile up keys.
            env_line = shlex.quote(f"{provider.key_env}={provider.api_key}")
            parts.append(
                f"f=\"$({hermes} config env-path)\" && touch \"$f\""
                f" && sed -i '/^{provider.key_env}=/d' \"$f\" && printf '%s\\n' {env_line} >> \"$f\""
            )
        parts.append(f"mkdir -p {workdir}")
        if want_files:
            parts.append(f"rm -rf {workdir}/outputs && mkdir -p {workdir}/outputs")
        # Last, so its line is the last of stdout: the folder as the agent must
        # name it, since a profile's terminal has a home of its own.
        parts.append(f"cd {workdir} && pwd")
        out = inst.exec(" && ".join(parts) + " < /dev/null")
        if out.get("exit_code") != 0:
            said = (out.get("stderr") or out.get("stdout") or "").strip()[-2000:]
            if provider:
                said = said.replace(provider.api_key, "<key>")
            raise AgentError(f"Preparing the agent failed: {said}", log)
        if profile not in ws.profiles:
            log.add("agent37", f"profile {profile} ready")
            ws.profiles.add(profile)
        done["steps"] = cap
        if new_provider:
            log.add("agent37", f"model: {provider.model} at {provider.base_url}")
            done["provider"] = provider
    lines = (out.get("stdout") or "").strip().splitlines()
    return lines[-1] if lines else workdir


def _stream_turn(inst: _Instance, task: str, profile: str, session: str, log: _Log) -> dict:
    """Send *task* as one streamed turn on *session*; log each tool call; return
    the terminal event.

    A session still busy with an earlier turn (a run abandoned on this side) has
    that turn cancelled and the send made again. A connection lost after the
    turn has started is a failure, never a resend, so a turn is not run twice.
    """
    for attempt in range(2):
        try:
            return _stream_once(inst, task, profile, session, log)
        except Agent37Error as exc:
            running = exc.err.get("response_id")
            if exc.code != "session_busy" or attempt or not running:
                raise
            log.add("agent37", f"session busy with {running}; cancelling it")
            inst._agent("POST", f"/v1/responses/{running}/cancel")
            time.sleep(2)
    raise AssertionError("unreachable")


def _stream_once(inst: _Instance, task: str, profile: str, session: str, log: _Log) -> dict:
    body = {"input": task, "profile": profile, "session_id": session, "stream": True}
    timeout = httpx.Timeout(connect=30, read=240, write=60, pool=30)
    deadline = time.time() + TIMEOUT_SECONDS
    response_id = None
    try:
        with inst.client.stream(
            "POST", f"{inst.url}/v1/responses", json=body,
            headers={"X-Agent37-Key": inst.key}, timeout=timeout,
        ) as r:
            if r.status_code >= 400:
                r.read()
                _raise_for(r)
            event = None
            for line in r.iter_lines():
                if time.time() > deadline:
                    if response_id:
                        inst._agent("POST", f"/v1/responses/{response_id}/cancel")
                    raise AgentError(f"The agent ran past {TIMEOUT_SECONDS}s and was stopped.", log)
                if not line or line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[5:])
                except ValueError:
                    continue
                if event == "response.created":
                    response_id = data.get("id")
                elif event == "response.tool_call.started":
                    label = (data.get("label") or "").strip().replace("\n", " ")
                    log.add("tool", f"{data.get('tool')}" + (f": {label[:LABEL_CHARS]}" if label else ""))
                elif event == "response.tool_call.failed":
                    log.add("tool", f"{data.get('tool')} failed: {data.get('error') or ''}")
                elif event in ("response.completed", "response.failed"):
                    return {"event": event, **data}
    except httpx.TransportError as exc:
        if response_id:
            raise AgentError(f"Lost the agent's stream mid-turn ({response_id}): {exc}", log) from exc
        raise
    raise AgentError("The agent's stream ended without a result.", log)


def _collect_files(inst: _Instance, outputs: str, log: _Log) -> list[pbt.File]:
    """The files the agent saved in *outputs* this run, as pbt files.

    Subfolders are flattened into the name (`plots/a.png` → `plots_a.png`):
    pbt file names never carry a path. A bullet that promised files and saved
    none fails, because everything downstream is waiting to be handed them.
    """
    listing = inst.exec(f"cd {outputs} && find . -type f -printf '%s\\t%P\\n'").get("stdout", "")
    found = []
    for line in listing.splitlines():
        size, _, rel = line.partition("\t")
        if rel and size.isdigit():
            found.append((rel, int(size)))
    found.sort()
    if not found:
        raise AgentError(
            f"This bullet is set to produce files, but the agent saved none in {outputs}/.", log
        )

    files: list[pbt.File] = []
    taken: set[str] = set()
    total = 0
    for rel, size in found:
        if len(files) >= MAX_FILES or total + size > MAX_FILES_BYTES:
            log.add("files", f"skipped {rel} ({size} bytes): over the limit of {MAX_FILES} files / {MAX_FILES_BYTES} bytes")
            continue
        data = inst._agent("GET", "/v1/files/content", params={"path": f"{outputs}/{rel}"}, timeout=300).content
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


def _answer(text: str, json_answer: bool, log: _Log) -> Any:
    text = text.strip()
    if not text:
        raise AgentError("The agent finished without an answer.", log)
    if not json_answer:
        return text[:MAX_OUTPUT_CHARS]
    try:
        return pbt_json(text)
    except ValueError as exc:
        raise AgentError(f"The agent's answer is not valid JSON: {exc}", log) from exc


def _run_instance(
    task: str,
    bullet: str,
    provider: _Provider | None,
    steps: int,
    json_answer: bool,
    on_line: Callable[[str], None] | None = None,
    want_files: bool = False,
    agent37_key: str | None = None,
    user: str = "",
    want_browser: bool = False,
    want_public: bool = False,
) -> dict:
    """Run the agent on *task* in the user's profile and the bullet's session.
    Blocking — call it off the loop."""
    key = agent37_key or os.environ[API_KEY_ENV]
    ws = _workspace(key)
    log = _Log(on_line)
    log.add("agent37", "on your own Agent37 workspace" if agent37_key else "on the shared Agent37 instance")
    profile = profile_for(user or "anonymous")
    session = session_id(bullet)
    outputs = ""
    files: list[pbt.File] = []

    def turn(inst: _Instance) -> dict:
        nonlocal outputs
        for attempt in range(2):
            workdir = _prepare(inst, ws, profile, session, steps, provider, want_files, log)
            outputs = f"{workdir}/outputs"
            text = task + FINISHING.format(steps=steps, workdir=workdir)
            if want_files:
                text += FILES_NOTE.format(outputs=outputs)
            if want_public:
                port, url = _public_url(inst, session)
                log.add("public", f"port {port} is public at {url}")
                text += PUBLIC_NOTE.format(port=port, url=url)
            if want_browser and not attempt:
                # Before the turn, so the person can watch it from the start.
                try:
                    log.add("desktop", f"watch the agent's screen: {_desktop_url(inst)}")
                except Exception as exc:  # noqa: BLE001 — the run does not need the view
                    log.add("desktop", f"no desktop link: {exc}")
            log.add(
                "agent",
                f"profile {profile}, session {session}: starting (up to {steps} steps), task of {len(task)} chars",
            )
            try:
                return _stream_turn(inst, text, profile, session, log)
            except Agent37Error as exc:
                # The profile went with an instance rebuilt behind this server's
                # back: make it again and send once more.
                if exc.code != "profile_not_found" or attempt:
                    raise
                log.add("agent37", f"profile {profile} not found; creating it")
                with ws.config_lock:
                    ws.profiles.discard(profile)
                    ws.configured.pop(profile, None)
        raise AssertionError("unreachable")

    with httpx.Client(timeout=60) as client:
        # Send first; bring the instance up only when the send cannot reach it.
        inst = None
        done = None
        if ws.shared is None:
            with ws.up_lock:
                if ws.shared is None and (found := _find(client, key)):
                    ws.shared = {"id": found["id"], "url": found["url"]}
        if ws.shared is not None:
            inst = _Instance(client, key, ws.shared)
            try:
                done = turn(inst)
            except Exception as exc:  # noqa: BLE001 — sorted by `_unreachable`
                if not _unreachable(exc):
                    raise
                log.add("agent37", f"instance not reachable ({exc}); bringing it up")
                ws.shared = None
        if done is None:
            inst = _bring_up(client, key, ws, log)
            done = turn(inst)

        if done["event"] == "response.failed":
            err = done.get("error") or {}
            raise AgentError(f"The agent failed: {err.get('code', '')} {err.get('message', '')}".strip(), log)
        usage = done.get("usage") or {}
        log.add(
            "agent",
            "finished"
            + (f", {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out tokens" if usage else "")
            + (f", ${usage['cost_usd']:.4f}" if usage.get("cost_usd") else ""),
        )
        answer = _answer(done.get("output_text") or "", json_answer, log)
        if want_files:
            files = _collect_files(inst, outputs, log)

    out: dict[str, Any] = {"output": answer}
    if want_files:
        out["files"] = files
    return {**out, "logs": log.lines(), "run_time": round(time.time() - log.t0, 1)}


async def _run(
    task: str,
    bullet: str,
    provider: _Provider | None,
    steps: int,
    json_answer: bool,
    on_line: Callable[[str], None] | None = None,
    want_files: bool = False,
    agent37_key: str | None = None,
    user: str = "",
    want_browser: bool = False,
    want_public: bool = False,
) -> dict:
    if not (agent37_key or enabled()):
        raise RuntimeError(
            "Preserving an agent's disk needs Agent37: add your own Agent37 key in Settings,"
            f" or set {API_KEY_ENV} on the server."
        )
    if not task.strip():
        raise RuntimeError("An agent bullet needs a task — its text, or something from below it.")
    return await asyncio.to_thread(
        _run_instance, task, bullet, provider, steps, json_answer, on_line, want_files, agent37_key, user,
        want_browser, want_public,
    )


@pbt.model_kind(MODEL_TYPE, config_keys={SESSION_KEY, STEPS_KEY, FILES_KEY, BROWSER_KEY, PUBLIC_KEY})
async def execute(rendered: str, call: pbt.ModelCall) -> dict:
    """Hand the rendered bullet to the Hermes agent, in the bullet's session on the shared instance.

    Returns ``{"output", "logs", "run_time"}``, like `agent_exec.execute`.
    """
    session = str(call.spec.config.get(SESSION_KEY) or call.spec.name)
    run = agent_exec._settings.get()
    try:
        own = int(call.spec.config.get(STEPS_KEY) or 0)
    except (TypeError, ValueError):
        own = 0
    steps = agent_exec.steps_for(own) if own > 0 else run.steps
    provider = _provider(run.provider, run.api_key, run.model)
    name = call.spec.name
    on_line = (lambda line: run.on_log(name, line)) if run.on_log else None
    json_answer = call.spec.output_format == "json"
    want_files = bool(call.spec.config.get(FILES_KEY))
    want_browser = bool(call.spec.config.get(BROWSER_KEY))
    want_public = bool(call.spec.config.get(PUBLIC_KEY))
    raw = await call.compute(
        rendered,
        compute=lambda: _run(
            rendered.strip(), session, provider, steps, json_answer, on_line, want_files,
            run.agent37_key, run.user, want_browser, want_public,
        ),
    )
    return raw if isinstance(raw, dict) else json.loads(raw)
