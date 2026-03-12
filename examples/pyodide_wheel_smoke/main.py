from __future__ import annotations

import json
import traceback

from js import document

status_el = document.getElementById("status")
output_el = document.getElementById("output")


def set_status(message: str, ok: bool | None = None) -> None:
    status_el.textContent = message
    if ok is True:
        status_el.className = "status ok"
    elif ok is False:
        status_el.className = "status fail"
    else:
        status_el.className = "status"


def llm_stub(prompt: str, **kwargs) -> str:
    return f"LLM::{prompt.strip()}"


try:
    import pbt
    from pbt.storage import MemoryStorageBackend

    models = {
        "topic": "Return exactly the word browser",
        "summary": '{{ skip_and_set_to_value(ref("topic")) }}',
    }

    results = pbt.run(
        models_from_dict=models,
        llm_call=llm_stub,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
    )

    output_el.textContent = json.dumps(
        {
            "pbt_version": pbt.__version__,
            "results": results,
            "expected_summary": "LLM::Return exactly the word browser",
            "passed": results.get("summary") == "LLM::Return exactly the word browser",
        },
        indent=2,
        sort_keys=True,
    )
    set_status("Wheel import and models_from_dict smoke test passed.", True)
except Exception:
    output_el.textContent = traceback.format_exc()
    set_status("Wheel smoke test failed.", False)
