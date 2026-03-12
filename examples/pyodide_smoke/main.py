from __future__ import annotations

from types import ModuleType
from pyodide.ffi import create_proxy
from js import document
import json
import traceback

status_el = document.getElementById("status")
output_el = document.getElementById("output")
button_el = document.getElementById("run-model")
model_source_el = document.getElementById("model-source")
button_proxy = None


def set_status(message: str, ok: bool | None = None) -> None:
    status_el.textContent = message
    if ok is True:
        status_el.className = "status ok"
    elif ok is False:
        status_el.className = "status fail"
    else:
        status_el.className = "status"


def llm_stub(prompt: str, **kwargs) -> str:
    return f"UNUSED::{prompt}"


def run_model(event=None) -> None:
    import pbt
    from pbt.storage import MemoryStorageBackend

    button_el.disabled = True
    set_status("Running inline model through pbt...", None)

    try:
        model_source = model_source_el.value
        models = {
            "modelinclude": model_source,
            "browser_result": (
                "{% set value = ref('modelinclude') %}\n"
                "{{ skip_and_set_to_value(value if value else '<empty string>') }}\n"
            ),
        }
        results = pbt.run(
            models_from_dict=models,
            llm_call=llm_stub,
            verbose=False,
            storage_backend=MemoryStorageBackend(),
        )
        output_el.textContent = json.dumps(
            {
                "submitted_model": "modelinclude",
                "model_source": model_source,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        set_status("pbt.run completed successfully.", True)
    except Exception:
        output_el.textContent = traceback.format_exc()
        set_status("pbt.run failed.", False)
    finally:
        button_el.disabled = False


try:
    import pbt

    button_proxy = create_proxy(run_model)
    button_el.addEventListener("click", button_proxy)
    button_el.disabled = False
    output_el.textContent = json.dumps(
        {
            "pbt_version": pbt.__version__,
            "ready": True,
            "message": "PyScript loaded local pbt files. Click the button to run models_from_dict.",
        },
        indent=2,
    )
    set_status("Local pbt package loaded successfully.", True)
except Exception:
    output_el.textContent = traceback.format_exc()
    set_status("Failed to load local pbt package.", False)
