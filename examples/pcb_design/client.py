import os
import random
import time

from google import genai
from google.genai import errors

#: Transient errors worth retrying: overloaded (503) and rate-limited (429).
#: An each= model sends one request per part at once, which a busy endpoint
#: rejects more often than single calls.
_RETRY_CODES = {429, 503}


def llm_call(prompt: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
    for attempt in range(6):
        try:
            return client.models.generate_content(model=model, contents=prompt).text
        except errors.APIError as exc:
            if exc.code not in _RETRY_CODES or attempt == 5:
                raise
            time.sleep(2 ** attempt + random.random())
