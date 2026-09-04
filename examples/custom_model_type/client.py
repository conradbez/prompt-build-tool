"""LLM backend + a custom model type for this project.

pbt imports this file before it reads models/, so any model type registered
here is available to every .prompt file.
"""

import os

import pbt
from google import genai


# ---------------------------------------------------------------------------
# A custom model type
# ---------------------------------------------------------------------------
# Use it from a .prompt file with:
#     {{ config(model_type="shout", suffix="!") }}
#
# config_keys tells pbt that "suffix" is a real option, not a typo.

@pbt.model_type("shout", config_keys={"suffix"})
class Shout(pbt.BaseModelType):
    async def execute(self, spec, ctx):
        rendered, state = ctx.render(spec)
        output = await ctx.call_llm(rendered, spec, state)
        return output.upper() + spec.config.get("suffix", "")


# ---------------------------------------------------------------------------
# The LLM backend
# ---------------------------------------------------------------------------

def llm_call(prompt: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        contents=prompt,
    ).text
