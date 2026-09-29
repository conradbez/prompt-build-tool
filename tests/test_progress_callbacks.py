"""on_model_start / on_model_done callbacks on pbt.run() and pbt.async_run()."""

from __future__ import annotations

import asyncio

import pbt
from pbt.storage import MemoryStorageBackend

MODELS = {
    "topic": "Name one topic.",
    "outline": "Outline {{ ref('topic') }}.",
    "article": "Write about {{ ref('outline') }}.",
}


def _events(llm_call, models=MODELS) -> tuple[list[tuple[str, str]], dict]:
    events: list[tuple[str, str]] = []
    results = pbt.run(
        models_from_dict=models,
        llm_call=llm_call,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
        on_model_start=lambda name: events.append(("start", name)),
        on_model_done=lambda r: events.append((r.status, r.model_name)),
    )
    return events, results


def test_callbacks_fire_in_dependency_order() -> None:
    events, _ = _events(lambda prompt: "ok")
    assert events == [
        ("start", "topic"), ("success", "topic"),
        ("start", "outline"), ("success", "outline"),
        ("start", "article"), ("success", "article"),
    ]


def test_done_reports_error_and_skipped() -> None:
    def llm(prompt: str) -> str:
        if prompt.startswith("Outline"):
            raise RuntimeError("boom")
        return "ok"

    events, _ = _events(llm)
    assert ("error", "outline") in events
    # article never starts; it is reported as skipped only.
    assert ("start", "article") not in events
    assert ("skipped", "article") in events


def test_done_receives_model_run_result() -> None:
    seen: list[pbt.ModelRunResult] = []
    pbt.run(
        models_from_dict={"topic": "Name one topic."},
        llm_call=lambda prompt: "hello",
        verbose=False,
        storage_backend=MemoryStorageBackend(),
        on_model_done=seen.append,
    )
    assert len(seen) == 1
    assert isinstance(seen[0], pbt.ModelRunResult)
    assert seen[0].llm_output == "hello"


def test_async_run_accepts_callbacks() -> None:
    started: list[str] = []

    async def llm(prompt: str) -> str:
        return "ok"

    asyncio.run(pbt.async_run(
        models_from_dict=MODELS,
        llm_call=llm,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
        on_model_start=started.append,
    ))
    assert started == ["topic", "outline", "article"]


def test_callbacks_fire_with_verbose_logging() -> None:
    events: list[str] = []
    pbt.run(
        models_from_dict={"topic": "Name one topic."},
        llm_call=lambda prompt: "ok",
        verbose=True,
        storage_backend=MemoryStorageBackend(),
        on_model_start=events.append,
    )
    assert events == ["topic"]
