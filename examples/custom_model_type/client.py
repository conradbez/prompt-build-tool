"""LLM backend + a custom model kind for this project.

pbt imports this file before it reads models/, so any model kind registered
here is available to every .prompt file.
"""

import os

import pbt
from google import genai


# ---------------------------------------------------------------------------
# A custom model kind
# ---------------------------------------------------------------------------
# Use it from a .prompt file with:
#     {{ config(model_type="shout", suffix="!") }}
#
# pbt renders the template and hands the text to this function; `call.llm` is
# the cached LLM call, already bound to this model.
#
# config_keys tells pbt that "suffix" is a real option, not a typo.

@pbt.model_kind("shout", config_keys={"suffix"})
async def shout(rendered, call):
    response = await call.llm(rendered)
    return response.upper() + call.spec.config.get("suffix", "")


# ---------------------------------------------------------------------------
# The LLM backend
# ---------------------------------------------------------------------------

def llm_call(prompt: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        contents=prompt,
    ).text
