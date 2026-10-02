"""execute_python: ref() in the code is part of the cache key, and yields the each= item."""

from __future__ import annotations

import json

from pbt.storage import MemoryStorageBackend
from tests.test_each_fanout import run_models

CALC = '{{ config(model_type="execute_python") }}\na = ref("arch")\noutput = len(a["paths"])\n'


def llm_with(n):
    return lambda prompt, config=None: json.dumps({"paths": list(range(n))})


def test_python_result_tracks_upstream_and_still_caches():
    storage = MemoryStorageBackend()
    models = {
        "calc": CALC,
        "arch": '{{ config(output_format="json") }}\nGive JSON for {{ promptdata("v") }}.',
    }
    values, cached = [], []
    for n in (2, 5, 5):
        _, _, r = run_models(models, storage=storage, llm_call=llm_with(n), promptdata={"v": n})
        values.append(r["calc"].value)
        cached.append(r["calc"].cached)
    assert values == ["2", "5", "5"]
    assert cached == [False, False, True]


def test_python_ref_yields_the_each_item():
    _, _, r = run_models({
        "items": '{{ config(output_format="json") }}\nList.',
        "each": '{{ config(model_type="execute_python", each="items") }}\noutput = ref("items") * 2\n',
    }, llm_call=lambda p, config=None: json.dumps(["a", "b"]))
    assert r["each"].value == ["aa", "bb"]
