"""The agent inside an agent bullet's sandbox: OpenCode, run headless.

Usage: run_agent.py <task file> <result file> [MCP server command…]

The task arrives as a file rather than an argument because it is a rendered
bullet — upstream answers and all — and has no business being limited by argv.
It reaches OpenCode on stdin. The result is written as JSON to *result file*.

OpenCode speaks MCP itself: the server whose command follows the result file
is launched over stdio, held open for the whole run (so its state carries from
one call to the next), and its tools reach the model as `mcp_<tool>`, next to
OpenCode's own bash, read, edit and the rest. Images a tool returns are shown
to the model.

Model and key come from the environment (`AGENT_MODEL`, as OpenCode's
`provider/model`, and the provider's own key variable, which OpenCode reads) —
set by `agent_exec.py` per run.
"""
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORKDIR = "/root"
# OpenCode finds its project, and so its config, by $PWD rather than the
# process's actual directory — an inherited PWD would lose the MCP server.
# Its websearch tool is only offered on OpenCode's own provider unless Exa is
# switched on; Exa's hosted endpoint needs no key.
OPENCODE_ENV = {**os.environ, "PWD": WORKDIR, "OPENCODE_ENABLE_EXA": "1"}
# The MCP server's name in OpenCode's config, which prefixes its tools.
MCP_NAME = "mcp"
STEP_LIMIT = int(os.environ.get("AGENT_STEP_LIMIT", "30"))
# OpenCode providers whose API refuses a request ending on the model's message.
NO_TRAILING_ASSISTANT = ("google/",)
# How those providers refuse it. OpenCode sends one not only at its step limit
# but whenever a step ends with neither an answer nor a tool call (finish
# "unknown", as a malformed function call gives): it asks again, unchanged.
TRAILING_MODEL_TURN = "ending with a model turn"
# So the run is resumed, up to this many times, with a user message after it.
RESUMES = 2
NUDGE = "Your last reply had neither a tool call nor an answer. Carry on with the task."
# A provider's rate limit or quota error. The run waits and is resumed, up to
# QUOTA_RETRIES times, the wait doubling from QUOTA_WAIT_START to at most
# QUOTA_WAIT_MAX seconds.
QUOTA_ERROR = re.compile(r"\b429\b|RESOURCE_EXHAUSTED|quota|rate.?limit|too many requests", re.I)
QUOTA_RETRIES = int(os.environ.get("AGENT_QUOTA_RETRIES", "6"))
QUOTA_WAIT_START = 5
QUOTA_WAIT_MAX = 30
QUOTA_NUDGE = "Carry on with the task."
# Seconds to wait after each model reply before the next call; 0 for none.
BACKOFF_SECONDS = float(os.environ.get("AGENT_BACKOFF_SECONDS") or 0)
# Where each OpenCode provider's API lives, for the back-off proxy to forward to.
UPSTREAMS = {
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
# Not forwarded either way: hop-by-hop, or set again for the new connection.
# The body is asked for uncompressed, so it streams through as it arrives.
_HOP_HEADERS = {"host", "connection", "keep-alive", "transfer-encoding", "content-length", "accept-encoding", "content-encoding"}

# A run stopped at its step limit is resumed once with this, and given one step
# to answer in: without it a model that kept calling tools (Gemini is never told
# tools are off, see `_config`) ends with nothing at all.
WRAP_UP = (
    "You have used all your steps. Do not call any more tools. Reply now with "
    "your final answer, from what you have so far; say what is unfinished."
)

FINISHING = """

---

## How to finish

You work in a Linux sandbox; your working directory is /root. Python 3.12,
`uv`/`uvx`, Node (`npx`), git and curl are installed.
{mcp}
You have about {steps} steps (one per reply of yours); leave the last of them
for the answer rather than running out mid-task.
Your final message is what the person reads, so make it the answer itself —
not a description of what you did or where you put it.
"""

FILES_NOTE = """
This task must produce files. Save every file the person should get — images,
documents, data — in {outputs}/ (it exists already). They are handed on as
files, so save at least one; your final message is handed on as text beside
them, so say what each file is rather than pasting its contents.
"""

BROWSER_NOTE = """
A Chromium window is open on display {display}, and a person may be watching it
live — and may click or type in it themselves. Use that browser, not a new
one, so they see what you do: drive it over the DevTools protocol at {cdp}
(e.g. Playwright's `connect_over_cdp("{cdp}")`, reusing its open page). If a
page needs something only the person can give — a login, a captcha — say so
in plain words and wait for it.
"""

# Where files the task asks for are saved; `agent_exec.py` collects them.
OUTPUTS_DIR = "/root/outputs"

# A files run that ends (answered, or at its step limit) with nothing in
# OUTPUTS_DIR is resumed once with this, and given SAVE_STEPS steps, tools on:
# the no-tools wrap-up alone could never save a file.
SAVE_STEPS = 5
SAVE_FILES = (
    "You have not saved any files in {outputs}/, and this task must produce "
    "files. Save them there now, from what you have so far — within {steps} "
    "steps, the last of them your final answer, given in full again."
)

MCP_NOTE = """
Tools named `mcp_*` come from an MCP server that stays up for the whole run, so
state carries over from one call to the next. Images they return are shown to
you, and each one costs tokens for the rest of the run: ask for renders or
screenshots only at key steps.
"""

# How much of each step goes into the bullet's log. The log is for following
# what happened, not a transcript: the model saw everything, the log a summary.
THOUGHT_CHARS = 600
OUTPUT_CHARS = 1500

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Each log event is also printed as it happens, on a stdout line of its own
# behind this marker, so the host can relay the run live instead of reading the
# whole log from the result file once it is over.
LIVE_MARKER = "@@agent-event "


# Values of the sandbox's secrets (the model's API key). The agent can print
# them (`env`, `cat /proc/self/environ`), and what it prints lands in the log
# and the answer, so every event and the result file are scrubbed of them.
SECRETS = [
    v for k, v in os.environ.items()
    if re.search(r"KEY|TOKEN|SECRET|PASSWORD", k) and len(v) >= 8
]


def _redact(text: str) -> str:
    for secret in SECRETS:
        text = text.replace(secret, "[redacted]")
    return text


def _record(events: list[dict], event: dict) -> None:
    event = {**event, "message": _redact(str(event.get("message", "")))}
    events.append(event)
    # One write, so a line from the back-off proxy's thread never splits another.
    sys.stdout.write(LIVE_MARKER + json.dumps(event) + "\n")
    sys.stdout.flush()


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + f"… [{len(text) - limit} more chars]"


def _config(mcp_command: list[str], base_url: str | None = None) -> dict:
    config: dict = {
        "$schema": "https://opencode.ai/config.json",
        "model": os.environ["AGENT_MODEL"],
        "autoupdate": False,
        "share": "disabled",
        "snapshot": False,
        # No one is there to answer a permission prompt.
        "permission": {"*": "allow"},
    }
    # Past this many steps OpenCode tells the model to stop using tools and
    # answer — by ending the request on an assistant message, which Gemini
    # refuses ("Requests ending with a model turn are not supported"). There
    # the model gets only the budget in FINISHING. Either way the limit itself
    # is enforced by `_read_run`, which stops a run that goes past it.
    if not config["model"].startswith(NO_TRAILING_ASSISTANT):
        config["agent"] = {"build": {"steps": STEP_LIMIT}}
    # The back-off proxy stands in for the provider's API.
    if base_url:
        provider = config["model"].split("/", 1)[0]
        config["provider"] = {provider: {"options": {"baseURL": base_url}}}
    if mcp_command:
        config["mcp"] = {
            MCP_NAME: {
                "type": "local",
                "command": mcp_command,
                # Per request. OpenCode's default is 5s; renders and exports are slow.
                "timeout": int(os.environ.get("AGENT_MCP_TIMEOUT_MS", "300000")),
            }
        }
    return config


def _backoff_proxy(upstream: str, wait: float, events: list[dict]) -> str:
    """Start a local proxy to *upstream* that paces model calls. Its base URL.

    Each call (a POST) waits until *wait* seconds have passed since the last
    one finished, and calls go one at a time — so a provider's rate limit sees
    a gap between every step, and between OpenCode's own retries too. Pausing
    OpenCode between its events instead could not promise that: by the time a
    step's end is read, the next request may already be on its way.

    The response is relayed as it arrives, so streaming is unaffected.
    """
    target = urllib.parse.urlsplit(upstream)
    lock = threading.Lock()
    last_done = [0.0]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # quiet: stderr is OpenCode's
            pass

        def _forward(self, pace: bool) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_HEADERS}
            headers["Accept-Encoding"] = "identity"
            with lock if pace else _nolock:
                if pace and last_done[0]:
                    left = last_done[0] + wait - time.time()
                    if left > 0:
                        _record(events, {"ts": time.time(), "source": "agent", "message": f"backing off {left:.1f}s before the next model call"})
                        time.sleep(left)
                conn = http.client.HTTPSConnection(target.netloc, timeout=600)
                try:
                    conn.request(self.command, target.path + self.path, body=body or None, headers=headers)
                    resp = conn.getresponse()
                    self.send_response(resp.status, resp.reason)
                    for k, v in resp.getheaders():
                        if k.lower() not in _HOP_HEADERS:
                            self.send_header(k, v)
                    # HTTP/1.0 and no length: the body ends when the connection does.
                    self.send_header("Connection", "close")
                    self.end_headers()
                    while chunk := resp.read1(65536):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                finally:
                    conn.close()
                    if pace:
                        last_done[0] = time.time()

        def do_POST(self) -> None:
            self._forward(pace=True)

        def do_GET(self) -> None:
            self._forward(pace=False)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


class _NoLock:
    def __enter__(self) -> None:
        pass

    def __exit__(self, *exc) -> None:
        pass


_nolock = _NoLock()


def _check_mcp(events: list[dict]) -> str | None:
    """Start the MCP server once to see that it comes up. The problem, or None.

    OpenCode would otherwise carry on without a server that failed and leave
    the agent to find out it has no tools.
    """
    started = time.time()
    try:
        p = subprocess.run(
            ["opencode", "mcp", "list"], cwd=WORKDIR, env=OPENCODE_ENV, capture_output=True, text=True,
            timeout=int(os.environ.get("AGENT_MCP_START_SECONDS", "180")),
        )
    except subprocess.TimeoutExpired:
        return "it did not come up in time"
    # A status line (`✓ mcp connected`, `✗ mcp failed`), then its details, each
    # on a line of its own under a `|` gutter.
    lines = _ANSI.sub("", p.stdout + p.stderr).splitlines()
    for i, line in enumerate(lines):
        status = re.search(rf"([✓✗])\s+{MCP_NAME}\s+(\S+)", line)
        if not status:
            continue
        if status.group(1) == "✓":
            _record(events, {"ts": time.time(), "source": "mcp", "message": f"server {status.group(2)} in {time.time() - started:.1f}s"})
            return None
        detail = []
        for more in lines[i + 1:]:
            if not more.startswith("|") or not more.strip(" |"):
                break
            detail.append(more.strip(" |"))
        return status.group(2) + (": " + "\n".join(detail) if detail else "")
    return "\n".join(lines).strip() or f"opencode mcp list exited {p.returncode}"


def _server_stderr(command: list[str]) -> str:
    """What the server prints when started on its own, for one that failed.

    OpenCode only reports that the connection closed; why (a package that is
    not on the index, a missing library) is on the server's stderr. With stdin
    closed a working server exits at once, and a broken one has already said
    what is wrong.
    """
    try:
        p = subprocess.run(command, cwd=WORKDIR, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
        # Both streams: a launcher like uv says "Installed N packages" on
        # stderr, and a server that exits at once often explains on stdout.
        return "\n".join(s.strip() for s in (p.stderr, p.stdout) if s.strip())[-2000:]
    except subprocess.TimeoutExpired as e:
        said = e.stderr or ""
        return (said.decode(errors="replace") if isinstance(said, bytes) else said).strip()[-2000:]
    except OSError as e:
        return str(e)


def _tool_event(part: dict, step: int, ts: float) -> dict:
    tool = part.get("tool", "?")
    state = part.get("state") or {}
    args = state.get("input") or {}
    call = f"$ {args['command']}" if tool == "bash" and "command" in args else f"{tool} {json.dumps(args)}"
    body = state.get("output") if state.get("status") == "completed" else state.get("error")
    images = sum(1 for a in state.get("attachments") or [] if str(a.get("mime", "")).startswith("image/"))
    message = f"step {step} {_clip(call, OUTPUT_CHARS)} -> {state.get('status', '?')}"
    if body:
        message += "\n" + _clip(str(body), OUTPUT_CHARS)
    if images:
        message += f"\n[{images} image(s) shown to the model]"
    source = "mcp" if tool.startswith(MCP_NAME + "_") else "tool"
    started = (state.get("time") or {}).get("start")
    return {"ts": started / 1000 if started else ts, "source": source, "message": message}


def _read_run(lines, result: dict, limit: int | None = None, stop=None, until: int | None = None) -> None:
    """Fold OpenCode's JSON event stream into *result*: the log, the answer,
    steps and cost.

    The answer is the text of the last step, when that step ended the turn
    rather than asking for tools. *lines* may be a live pipe: each event is
    logged as its line arrives.

    *limit* is enforced here, not trusted to OpenCode. Its own `steps` only
    tells the model that tools are off while still sending them — a model
    that carries on calling them is not stopped — and it is not set at all for
    Gemini. The step after the limit is the one the model was told to answer
    in; a step past that (or past *until*, when given) calls *stop*, and the
    run ends without an answer.

    Steps and cost carry on from *result*'s, so a resumed run counts on.
    """
    events = result["events"]
    step, cost = result.get("steps", 0), result.get("cost", 0.0)
    over = False
    texts: dict[str, list[str]] = {}
    last_finish: dict = {}
    errors = []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        part = e.get("part") or {}
        ts = e.get("timestamp", time.time() * 1000) / 1000
        kind = e.get("type")
        if kind == "step_start":
            step += 1
            if limit and step > (until or limit + 1):
                over = True
                _record(events, {"ts": ts, "source": "agent", "message": f"stopped: step limit of {limit} reached without an answer"})
                if stop:
                    stop()
                break
        elif kind == "text" and part.get("text", "").strip():
            texts.setdefault(part.get("messageID", ""), []).append(part["text"])
            at = (part.get("time") or {}).get("start")
            _record(events, {"ts": at / 1000 if at else ts, "source": "agent", "message": f"step {step}: {_clip(part['text'], THOUGHT_CHARS)}"})
        elif kind == "tool_use":
            _record(events, _tool_event(part, step, ts))
        elif kind == "step_finish":
            cost += float(part.get("cost") or 0)
            last_finish = part
        elif kind == "error":
            err = e.get("error") or {}
            errors.append(f"{err.get('name', 'Error')}: {(err.get('data') or {}).get('message', '')}".strip())
            _record(events, {"ts": ts, "source": "agent", "message": f"error {errors[-1]}"})

    answer = "\n".join(texts.get(last_finish.get("messageID", ""), [])).strip()
    said = [t for group in texts.values() for t in group]
    result.update(steps=step, cost=cost, last_message=said[-1] if said else result.get("last_message", ""))
    if over:
        result.update(exit_status=f"StepLimit {limit}", steps=step - 1)
    elif errors:
        result.update(exit_status="Error", error="\n".join(errors))
    elif last_finish.get("reason") == "stop" and answer:
        result.update(exit_status="Answered", submission=answer)
    else:
        result.update(exit_status=f"Stopped ({last_finish.get('reason') or 'no step finished'})")


def _stop(p: subprocess.Popen) -> None:
    """End OpenCode and everything it started."""
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _opencode(command: list[str], prompt: str, result: dict, until: int | None = None) -> tuple[int, str]:
    """Run OpenCode on *prompt*, folding its events into *result*. Its exit
    code and stderr.

    Read as it runs, not once it is done, so each step reaches the log live.
    stderr goes to a file: a pipe nobody reads can fill and stall it.
    """
    with tempfile.TemporaryFile("w+") as stderr:
        p = subprocess.Popen(
            command,
            cwd=WORKDIR, env=OPENCODE_ENV, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, bufsize=1,
            # Its own group, so a stop takes the MCP server it started too.
            start_new_session=True,
        )
        p.stdin.write(prompt)
        p.stdin.close()
        _read_run(p.stdout, result, limit=STEP_LIMIT, stop=lambda: _stop(p), until=until)
        code = p.wait()
        stderr.seek(0)
        return code, stderr.read()


def _saved_files() -> bool:
    """Whether the agent has saved anything in OUTPUTS_DIR, subfolders included."""
    return any(files for _, _, files in os.walk(OUTPUTS_DIR))


def main(task_path: str, result_path: str, mcp_command: list[str]) -> None:
    result: dict = {"started": time.time(), "model": os.environ["AGENT_MODEL"], "exit_status": "", "submission": "", "events": []}
    with open(task_path) as f:
        task = f.read()
    base_url = None
    if BACKOFF_SECONDS > 0:
        upstream = UPSTREAMS.get(result["model"].split("/", 1)[0])
        if upstream:
            base_url = _backoff_proxy(upstream, BACKOFF_SECONDS, result["events"])
        else:
            _record(result["events"], {"ts": time.time(), "source": "agent", "message": f"back-off ignored: no known API for {result['model']}"})
    with open(os.path.join(WORKDIR, "opencode.json"), "w") as f:
        json.dump(_config(mcp_command, base_url), f, indent=2)

    try:
        problem = _check_mcp(result["events"]) if mcp_command else None
        if problem:
            said = _server_stderr(mcp_command)
            result.update(exit_status="MCPServerFailed", steps=0, error=problem + (f"\nIt printed:\n{said}" if said else ""))
            return
        task += FINISHING.format(mcp=MCP_NOTE if mcp_command else "", steps=STEP_LIMIT)
        wants_files = os.environ.get("AGENT_OUTPUT_FILES") == "1"
        if wants_files:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            task += FILES_NOTE.format(outputs=OUTPUTS_DIR)
        if os.environ.get("AGENT_REMOTE_BROWSER") == "1":
            task += BROWSER_NOTE.format(display=os.environ.get("DISPLAY", ":99"), cdp=os.environ["AGENT_CDP_URL"])
        # --title skips a model call spent naming the session.
        command = ["opencode", "run", "--format", "json", "--auto", "--title", "agent bullet"]
        resumes = waits = 0
        while True:
            code, said = _opencode(command, task, result)
            error = result.get("error", "")
            if TRAILING_MODEL_TURN in error and resumes < RESUMES:
                resumes += 1
                _record(result["events"], {"ts": time.time(), "source": "agent", "message": "resuming: the last step had neither a tool call nor an answer"})
                nudge = NUDGE
            # The error may be an event or, when OpenCode gives up, only on stderr.
            elif QUOTA_ERROR.search(error or (said if code else "")) and waits < QUOTA_RETRIES:
                wait = min(QUOTA_WAIT_START * 2 ** waits, QUOTA_WAIT_MAX)
                waits += 1
                _record(result["events"], {"ts": time.time(), "source": "agent", "message": f"rate limited: waiting {wait}s, then resuming ({waits}/{QUOTA_RETRIES})"})
                time.sleep(wait)
                nudge = QUOTA_NUDGE
            else:
                break
            result.pop("error", None)
            command, task = ["opencode", "run", "--format", "json", "--auto", "--continue"], nudge
        continue_ = ["opencode", "run", "--format", "json", "--auto", "--continue"]
        ended = result["exit_status"] == "Answered" or result["exit_status"].startswith("StepLimit")
        if wants_files and ended and not _saved_files():
            _record(result["events"], {"ts": time.time(), "source": "agent", "message": f"resuming: no files in {OUTPUTS_DIR}/ yet, asking for them, up to {SAVE_STEPS} steps"})
            prompt = SAVE_FILES.format(outputs=OUTPUTS_DIR, steps=SAVE_STEPS)
            code, said = _opencode(continue_, prompt, result, until=result["steps"] + SAVE_STEPS)
        if result["exit_status"].startswith("StepLimit"):
            _record(result["events"], {"ts": time.time(), "source": "agent", "message": "resuming: asking for the answer, one step, no tools"})
            code, said = _opencode(continue_, WRAP_UP, result, until=result["steps"] + 1)
        stopped = result["exit_status"].startswith("StepLimit")
        if code and not stopped and not result.get("error") and result["exit_status"] != "Answered":
            result["error"] = f"opencode exited {code}:\n{said.strip()[-3000:]}"
    except Exception as e:  # noqa: BLE001 — reported back, not swallowed
        result.update(exit_status=type(e).__name__, error=str(e))
    finally:
        with open(result_path, "w") as f:
            f.write(_redact(json.dumps(result)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
