"""
Demonstrates tracking live progress while pbt runs.

pbt.run() and pbt.async_run() accept two optional callbacks:
  on_model_start(name)     — fired just before a model starts
  on_model_done(result)    — fired when a model finishes; result.status is
                             "success", "error" or "skipped"

Shows two usage patterns:
  1. Status board — keep a dict of every model's status, print it on each change
  2. Async event stream — push events onto an asyncio.Queue for a consumer
     (e.g. a websocket handler or UI) to read while the run is in progress

Uses a stub LLM, so no API key is needed.
"""

import asyncio
import os
import random
import time

import pbt
from pbt.storage import MemoryStorageBackend

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "example_test_run", "models")
PROMPTFILES = {"style_guide": os.path.join(HERE, "example_test_run", "example_article.md")}


# ---------------------------------------------------------------------------
# Example 1 — status board: a dict of every model's status, updated live
# ---------------------------------------------------------------------------

def slow_llm(prompt: str, files: list[str] | None = None) -> str:
    """Stub that takes a moment, so progress is visible — replace with a real LLM call."""
    time.sleep(random.uniform(0.2, 0.6))
    return "no"  # also answers model_to_skip's yes/no question


def example_status_board():
    print("=== Example 1: status board ===")

    statuses: dict[str, str] = {}

    def print_board() -> None:
        board = "  ".join(f"{name}={status}" for name, status in statuses.items())
        print(f"  {board}")

    def on_start(name: str) -> None:
        statuses[name] = "running"
        print_board()

    def on_done(result: pbt.ModelRunResult) -> None:
        statuses[result.model_name] = result.status
        print_board()

    pbt.run(
        models_dir=MODELS_DIR,
        llm_call=slow_llm,
        promptfiles=PROMPTFILES,
        verbose=False,  # our board replaces pbt's own log
        storage_backend=MemoryStorageBackend(),  # keep run history out of the local DB
        on_model_start=on_start,
        on_model_done=on_done,
    )


# ---------------------------------------------------------------------------
# Example 2 — async event stream: consume events while the run is in progress
# ---------------------------------------------------------------------------

async def async_slow_llm(prompt: str, files: list[str] | None = None) -> str:
    """Async stub — an async llm_call keeps the event loop free for the consumer."""
    await asyncio.sleep(random.uniform(0.2, 0.6))
    return "no"


async def example_event_stream():
    print("\n=== Example 2: async event stream ===")

    events: asyncio.Queue = asyncio.Queue()

    async def consumer() -> None:
        # Stand-in for sending events to a browser, websocket, log service, ...
        while (event := await events.get()) is not None:
            kind, name, detail = event
            print(f"  [{kind:5s}] {name:15s} {detail}")

    consumer_task = asyncio.create_task(consumer())

    results = await pbt.async_run(
        models_dir=MODELS_DIR,
        llm_call=async_slow_llm,
        promptfiles=PROMPTFILES,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
        on_model_start=lambda name: events.put_nowait(("start", name, "")),
        on_model_done=lambda r: events.put_nowait(("done", r.model_name, r.status)),
    )

    await events.put(None)  # tell the consumer the run is over
    await consumer_task
    print(f"  finished: {len(results)} models")


# ---------------------------------------------------------------------------
# Run all examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_status_board()
    asyncio.run(example_event_stream())
