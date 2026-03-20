import mimetypes
import os
from google import genai

# Automatically picked up by `pbt` to run .prompt files.
# Optional kwargs passed by pbt when declared in the signature:
#   files  - list of open file objects attached to the prompt (via --promptdata key=@path)
#   config - dict of {{ config(...) }} options from the .prompt file (e.g. {"output_format": "json"})
def llm_call(prompt: str, files: list | None = None, config: dict | None = None) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    uploaded = []
    for f in (files or []):
        mime, _ = mimetypes.guess_type(f.name)
        uploaded.append(client.files.upload(
            file=f,
            config={"mime_type": mime or "text/plain"},
        ))
    contents = [prompt] + uploaded if uploaded else prompt
    return client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        contents=contents,
    ).text
