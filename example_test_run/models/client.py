import os

from google import genai

_DEFAULT_MODEL = "gemini-3-flash-preview"


def llm_call(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Obtain a key at https://aistudio.google.com/app/apikey "
            "and export it:\n\n  export GEMINI_API_KEY=your_key_here"
        )
    model_name = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)
    client = genai.Client(api_key=api_key)
    return client.models.generate_content(model=model_name, contents=prompt).text
