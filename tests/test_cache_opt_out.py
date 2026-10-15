"""config(cache=False) skips reading the prompt cache, but still writes it."""

from __future__ import annotations

import pbt
from pbt.storage import MemoryStorageBackend
from tests.test_each_fanout import run_models


class CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str, config=None) -> str:
        self.calls += 1
        return f"answer {self.calls}"


def _run(models: dict[str, str], llm, storage) -> dict:
    return run_models(models, storage=storage, llm_call=llm)[2]


def test_cache_false_always_calls_the_llm() -> None:
    storage, llm = MemoryStorageBackend(), CountingLLM()
    models = {"a": "{{ config(cache=False) }}\nHi"}

    _run(models, llm, storage)
    results = _run(models, llm, storage)

    assert llm.calls == 2
    assert results["a"].cached is False


def test_cache_false_writes_the_entry_a_cached_model_then_reads() -> None:
    """Toggling cache= does not change the key: the fresh result is what a later hit serves."""
    storage, llm = MemoryStorageBackend(), CountingLLM()

    cached = {"a": "{{ config(cache=True) }}\nHi"}
    _run(cached, llm, storage)
    _run({"a": "{{ config(cache=False) }}\nHi"}, llm, storage)
    results = _run(cached, llm, storage)

    assert llm.calls == 2
    assert results["a"].cached is True
    assert results["a"].value == "answer 2"


def test_cache_is_a_known_config_key(recwarn) -> None:
    from pbt.executor.graph import build_models_from_dict

    build_models_from_dict({"a": "{{ config(cache=False) }}\nHi"})
    assert [w for w in recwarn if issubclass(w.category, pbt.UnknownConfigKeyWarning)] == []
