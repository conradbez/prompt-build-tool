"""
Validation framework for promptmap.

After each model's LLM result is returned, promptmap looks for a matching Python
file in the validation/ directory (e.g. validation/topic.py for topic.prompt).

Each validation file must define::

    def validate(prompt: str, result: str) -> Any:
        ...

The return value of ``validate`` becomes the model's output and is passed
downstream via ``ref()``.  Return ``False`` (or raise) to mark the model
as an error.  Any other return value — including a transformed string, a
parsed dict, or ``True`` — replaces the LLM output.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable


def load_validators(validation_dir: str | Path) -> dict[str, Callable[[str, str], Any]]:
    """
    Discover all *.py files in *validation_dir* and load their ``validate``
    functions.

    Returns a dict mapping model_name → validate callable.
    Returns an empty dict if the directory does not exist.

    Raises
    ------
    AttributeError
        If a validation file exists but does not define ``validate``.
    """
    vdir = Path(validation_dir)
    if not vdir.exists():
        return {}

    validators: dict[str, Callable[[str, str], Any]] = {}

    for py_file in sorted(vdir.glob("*.py")):
        model_name = py_file.stem
        spec = importlib.util.spec_from_file_location(
            f"promptmap_validator_{model_name}", py_file
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"promptmap_validator_{model_name}"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "validate"):
            raise AttributeError(
                f"Validation file '{py_file}' must define a "
                f"'validate(prompt: str, result: str) -> Any' function."
            )

        validators[model_name] = module.validate

    return validators


def run_validator(
    model_name: str,
    validators: dict[str, Callable[[str, str], Any]],
    prompt: str,
    result: str,
) -> Any:
    """
    Run the validator for *model_name* if one exists, and return the output
    that should be stored and passed downstream.

    - If no validator exists for *model_name*, returns *result* unchanged.
    - If the validator returns ``False`` (or raises), raises ``ValueError``
      so the model is marked as an error.
    - Otherwise the validator's return value becomes the model's output.

    Parameters
    ----------
    model_name:
        Name of the model being validated.
    validators:
        Mapping loaded by ``load_validators()``.
    prompt:
        The fully-rendered prompt sent to the LLM.
    result:
        The raw LLM output string.

    Returns
    -------
    The final output to store and make available to downstream models via ref().
    """
    validator = validators.get(model_name)
    if validator is None:
        return result

    try:
        validated = validator(prompt, result)
    except Exception as exc:
        raise ValueError(
            f"Validator for '{model_name}' raised an exception: {exc}"
        ) from exc

    if validated is False:
        raise ValueError(
            f"Validator for '{model_name}' returned False — output did not pass validation."
        )

    return validated
