import os

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

# Picks up OPENROUTER_API_KEY (and AGENT37_API_KEY for the audio model) from
# the nearest .env above where you run promptmap.
load_dotenv(find_dotenv(usecwd=True))

MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-5.5")

# The agent37 model (audio) takes its LLM from the run settings, which default
# to Gemini. Point it at OpenRouter too.
try:
    from promptmap_extra_model_kinds import agent_exec

    agent_exec.use_run(agent_exec.RunSettings(
        provider="openrouter",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        model=MODEL,
    ))
except ImportError:
    pass  # no extras: promptmap stops on the agent37 model and says what to install


def llm_call(prompt: str) -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
