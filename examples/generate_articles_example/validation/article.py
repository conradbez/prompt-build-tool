def validate(prompt: str, result: str) -> str:
    """Article must be at least 200 characters and contain a markdown header."""
    didPassValidation = len(result) >= 200 and "#" in result
    if not didPassValidation:
        result = False
    return result
