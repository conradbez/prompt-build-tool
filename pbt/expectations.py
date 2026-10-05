"""Deterministic partial comparisons for YAML case expectations."""


def mismatches(actual, expected, path):
    """Compare selected mapping fields; match ref-bearing list items by ref.

    Other lists are positional and must have the expected length. Scalar
    comparisons are exact, with booleans kept distinct from numbers.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected a mapping, got {actual!r}"]
        errors = []
        for key, value in expected.items():
            child = f"{path}.{key}"
            if key not in actual:
                errors.append(f"{child}: missing field")
            else:
                errors.extend(mismatches(actual[key], value, child))
        return errors
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected a list, got {actual!r}"]
        errors = []
        if expected and all(isinstance(item, dict) and "ref" in item for item in expected):
            for item in expected:
                ref = item["ref"]
                child = f"{path}[ref={ref}]"
                matches = [v for v in actual if isinstance(v, dict) and v.get("ref") == ref]
                if len(matches) != 1:
                    errors.append(f"{child}: expected one matching item, got {len(matches)}")
                else:
                    errors.extend(mismatches(matches[0], item, child))
            return errors
        if len(actual) != len(expected):
            errors.append(f"{path}: expected {len(expected)} items, got {len(actual)}")
        for index, (value, wanted) in enumerate(zip(actual, expected)):
            errors.extend(mismatches(value, wanted, f"{path}[{index}]"))
        return errors
    if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        return [f"{path}: expected {expected!r}, got {actual!r}"]
    return []
