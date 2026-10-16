import os

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

# Picks up OPENROUTER_API_KEY (and AGENT37_API_KEY for the audio model) from
# the nearest .env above where you run pbt.
load_dotenv(find_dotenv(usecwd=True))


def llm_call(prompt: str) -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    resp = client.chat.completions.create(
        model=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-5.5"),
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
