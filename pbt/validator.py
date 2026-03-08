"""
Validation framework for pbt.

After each model's LLM result is returned, pbt looks for a matching Python
file in the validation/ directory (e.g. validation/topic.py for topic.prompt).

Each validation file must define::

    def validate(prompt: str, result: str) -> bool:
        ...

If ``validate`` returns ``False`` (or raises), the model is marked as an error.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable


def load_validators(validation_dir: str | Path) -> dict[str, Callable[[str, str], bool]]:
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

    validators: dict[str, Callable[[str, str], bool]] = {}

    for py_file in sorted(vdir.glob("*.py")):
        model_name = py_file.stem
        spec = importlib.util.spec_from_file_location(
            f"pbt_validator_{model_name}", py_file
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"pbt_validator_{model_name}"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "validate"):
            raise AttributeError(
                f"Validation file '{py_file}' must define a "
                f"'validate(prompt: str, result: str) -> bool' function."
            )

        validators[model_name] = module.validate

    return validators


def run_validator(
    model_name: str,
    validators: dict[str, Callable[[str, str], bool]],
    prompt: str,
    result: str,
) -> None:
    """
    Run the validator for *model_name* if one exists.

    Raises
    ------
    ValueError
        If the validator returns False or raises an exception.
    """
    validator = validators.get(model_name)
    if validator is None:
        return

    try:
        passed = validator(prompt, result)
    except Exception as exc:
        raise ValueError(
            f"Validator for '{model_name}' raised an exception: {exc}"
        ) from exc

    if not passed:
        raise ValueError(
            f"Validator for '{model_name}' returned False — output did not pass validation."
        )
