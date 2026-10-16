# Agent37 — what the platform can do, and what the agent37 node uses

Reference for `agent37_exec.py`, the *preserve disk between runs* agent node.
Source: <https://www.agent37.com/docs> (full index: `/docs/llms.txt`) and the
dashboard Playground (`agent37.com/dashboard/cloud/playground`), which runs any
call below against a real instance and copies it as cURL.

## How the node maps onto Agent37

| pbt | Agent37 / Hermes | Value |
| - | - | - |
| Server key (`AGENT37_API_KEY`) or a user's key (Settings) | Workspace | one shared instance per key, named `pbt-shared` |
| pbt user (signed-in email, else browser session id) | Hermes **profile** | `u-` + `sha256(user)[:20]` |
| Node | Hermes **session** | `session_id = sha256(node id)[:32]` |
| Node's folder | dir on the instance's disk | `~/.hermes/profiles/<profile>/pbt/<session>/` |
| Node's *Create public URL* | public port | `20000 + int(session[:8], 16) % 10000`, label `pbt-<session[:24]>` |
| Node's *Create browser* | signed URL to the desktop | port 6901, `AGENT37_DESKTOP_TTL_SECONDS` |

Per run: `exec` (make profile if new, write its config, make the folder) →
`POST /v1/responses {profile, session_id, input, stream}` → collect files.
Instance is brought up only when that fails as unreachable. See the module
docstring for the lifecycle in full.

## Two planes, one key

| Plane | Base URL | Auth header |
| - | - | - |
| **Sandbox API** — manage instances | `https://api.agent37.com/v1` | `Authorization: Bearer sk_live_…` |
| **Agent API** — talk to the agent on one instance | `https://{instanceId}.agent37.app` | `X-Agent37-Key: sk_live_…` |

Any request to an instance URL wakes a sleeping instance (held up to ~3 min, then
`503 wake_timeout`). Sandbox API reads never reset the idle timer.

## Sandbox API — `api.agent37.com/v1`

### Instances — lifecycle

| Call | Endpoint | Used by the node |
| - | - | - |
| Create | `POST /instances` `{template, name, budget, auto_sleep, idle_timeout_seconds, env…}` | ✅ first run, with auto-sleep |
| List | `GET /instances` | ✅ find `pbt-shared` by name |
| Get | `GET /instances/{id}` | ✅ status polling |
| Edit | `PATCH /instances/{id}` — labels, metadata, env, auto-sleep | ✅ turns auto-sleep on |
| Start | `POST /instances/{id}/start` — start stopped, wake sleeping | ✅ when the send fails |
| Stop | `POST /instances/{id}/stop` — disk-only billing until `start` | — (auto-sleep instead) |
| Restart | `POST /instances/{id}/restart` — same image and files | — |
| Update | `POST /instances/{id}/update` — new template image, keeps `/home/node` | — |
| Resize | `POST /instances/{id}/resize` — CPU / memory / disk, same id and URLs | — |
| Fork | `POST /instances/{id}/fork` — independent copy | — |
| Delete | `DELETE /instances/{id}` | — (do it by hand to reset everyone) |

Statuses: `running`, `sleeping`, `stopped`, `deleting`, `deleted`, `failed`.
`409 try_again` while a sleep checkpoint is being written — retry with backoff.

### Auto-sleep

`auto_sleep: true` + `idle_timeout_seconds` (300–86400, default 900). No bytes on
any instance URL (or exec) for that long → checkpoint → `sleeping`, billed disk
only. Wake is transparent and takes seconds; after a long sleep it may boot fresh
(~2 min, files kept, memory lost), so poll `GET /v1/health` before sending work.

### Access

| Call | Endpoint | Used by the node |
| - | - | - |
| Exec | `POST /instances/{id}/exec` `{command}` → `{exit_code, stdout, stderr}`; output capped at 512 KB; wakes a sleeper | ✅ profile + config + folder setup, listing outputs |
| Signed URL | `POST /instances/{id}/signed-url` `{port, ttl_seconds}` | ✅ *Create browser*: noVNC desktop link (port 6901) in the log, before the turn |
| List public URLs | `GET /instances/{id}/public-ports` | ✅ find the bullet's port by label `pbt-<session>` |
| Create public URL | `POST /instances/{id}/public-ports` `{port, label}` — one port, permanent, no auth (webhooks, embeds); `DELETE …/public-ports/{port}`; max 50 per instance | ✅ *Create public URL*: port 20000–29999 picked by session id, told to the agent |
| SSH | your own keypair: ssh, scp, sftp, VS Code Remote | — |
| Logs / metrics | boot + runtime logs; CPU / memory / disk history | — |

### Budget — managed services (LLM router, Brave Search, Composio, Perflo)

| Call | Endpoint | Used by the node |
| - | - | - |
| Get budget | `GET /instances/{id}/budget` | — |
| Set budget | `PATCH /instances/{id}/budget` | set once at create (`AGENT37_BUDGET_MICROS`) |
| Top up | `POST /instances/{id}/budget/top-up` | — |
| Usage | `GET /instances/{id}/usage` → `{period, total_micros, by_integration}` | — |

Compute is billed separately, per minute awake, from the workspace wallet.

### Templates, backups, crons

| Area | What it gives | Used by the node |
| - | - | - |
| Templates | `GET /templates[/{name}]`; build from a Dockerfile in the cloud | ✅ `hermes-vnc-desktop` (see server/README.md) |
| Backups | `GET/POST /instances/{id}/backups`, `POST /instances/{id}/restore` | — |
| Crons | `/instances/{id}/crons` — a prompt on a schedule, per `profile`, fresh session per firing, wakes the instance; max 50 | — |
| Integrations | Composio: Gmail, Slack, Notion, 250+ apps, OAuth handled | — |
| Messaging | Telegram, WhatsApp, Slack, Discord… to the agent | — |

## Agent API — `{instanceId}.agent37.app`

| Call | Endpoint | Used by the node |
| - | - | - |
| Send a message | `POST /v1/responses` `{input, session_id, profile, model, provider, reasoning_effort, files, stream}` | ✅ one turn per run |
| Stream | SSE: `response.created`, `response.output_text.delta`, `response.tool_call.started/failed`, `response.completed/failed` | ✅ tool calls go to the run log |
| Reconnect | `GET /v1/responses/{id}/stream` — replays the turn (~30 min) | — (a dropped stream fails the run) |
| Cancel | `POST /v1/responses/{id}/cancel` | ✅ on timeout and on `session_busy` |
| Sessions | `GET/PATCH/DELETE /v1/sessions[/{id}]?profile=` — list, transcript, `active_response_id` | — |
| Files | `GET /v1/files?path=` list; `/v1/files/content` read; `/v1/files/archive` tar.gz a folder; write; `DELETE` (recursive, no undo) | ✅ `content` to collect outputs |
| Models | `GET /v1/models?profile=` | — |
| Health | `GET /v1/health` → `{healthy}` | ✅ after bring-up |

### Sessions and profiles

- **Session** = one conversation. Omit `session_id` and one is minted; send an
  unknown 32-hex id and Hermes starts a fresh thread under it — which is how the
  node gets a stable id per node without storing anything.
- One turn per session at a time: a second send is `409 session_busy` with
  `error.response_id`.
- **Profile** (Hermes only) = a named setup with its own `SOUL.md`, skills,
  memory, config and session store: `~/.hermes/profiles/<name>`. Lowercase
  letters, digits, `-`, `_`. Unknown → `404 profile_not_found`. Each runs as its
  own process, started on first request, stopped after 10 idle minutes.
  Created with `hermes profile create <name> [--clone]`; configured with
  `hermes --profile=<name> config set …`.
- Profiles separate config, memory and history — **not the disk**. They are one
  Linux user; an agent can read another profile's files by path.

## Errors the node branches on

| Code | Status | Node does |
| - | - | - |
| `container_unavailable` | 502 | instance stopped → start it, resend |
| `container_unreachable`, `upstream_unreachable` | 502 | bring up, resend |
| `wake_timeout`, `wake_failed`, `host_mesh_not_ready` | 503 | bring up, resend |
| `not_found` | 404 | instance gone → find / create, resend |
| `try_again` | 409 | bring up (start retries it with backoff) |
| `instance_limit_reached` | 409 | create waits for a free slot (`AGENT37_QUEUE_SECONDS`) |
| `profile_not_found` | 404 | create the profile, resend once |
| `session_busy` | 409 | cancel the running turn, resend once |
| `insufficient_balance`, `instance_suspended` | 402 | fail — top up the wallet |
| `upstream_timeout` | 504 | fail (15 min of silence) |

## Settings the node reads

| Env var | Default | Meaning |
| - | - | - |
| `AGENT37_API_KEY` | — | server's key; the shared instance. Users can set their own in Settings |
| `AGENT37_INSTANCE_NAME` | `pbt-shared` | the shared instance's name |
| `AGENT37_INSTANCE_ID` | `9c7c78hguk` | share this existing instance (server key only); wins over the name |
| `AGENT37_TEMPLATE` | `hermes-vnc-desktop` | template it is created from |
| `AGENT37_IDLE_SECONDS` | `600` | auto-sleep after this much idle |
| `AGENT37_BUDGET_MICROS` | `1000000` | managed-services cap per instance |
| `AGENT37_MANAGED_MODEL` | off | run on Agent37's model instead of the run's provider key |
| `AGENT37_QUEUE_SECONDS` / `AGENT37_START_SECONDS` / `AGENT_READY_SECONDS` | 900 / 900 / 240 | waits for a slot, a start, a healthy agent |
| `AGENT37_DESKTOP_TTL_SECONDS` | `300` | lifetime of the desktop link |
