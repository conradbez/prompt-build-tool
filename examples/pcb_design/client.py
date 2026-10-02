import os
import random
import time

from google import genai
from google.genai import errors, types

#: Transient errors worth retrying: overloaded (503) and rate-limited (429).
#: An each= model sends one request per part at once, which a busy endpoint
#: rejects more often than single calls.
_RETRY_CODES = {429, 503}


def llm_call(prompt: str, files: list | None = None) -> str:
    """Send *prompt*, plus any attached files (datasheet crops, notes, PDFs)."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    contents = [prompt, *(_part(f) for f in files or [])]
    for attempt in range(6):
        try:
            return client.models.generate_content(model=model, contents=contents).text
        except errors.APIError as exc:
            if exc.code not in _RETRY_CODES or attempt == 5:
                raise
            time.sleep(2 ** attempt + random.random())


def _part(handle) -> "types.Part":
    """An attached file as an inline part. Gemini reads text/plain, not markdown."""
    mime = getattr(handle, "mime", None) or "application/octet-stream"
    if mime.startswith("text/"):
        mime = "text/plain"
    return types.Part.from_bytes(data=handle.read(), mime_type=mime)
