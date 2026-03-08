"""
Demonstrates running pbt programmatically from Python.

Shows three usage patterns:
  1. Default run — uses models/client.py or built-in Gemini + models/rag.py
  2. Inline llm_call — swap in any LLM without touching the models directory
  3. Inline rag_call — provide a custom RAG function at call time
"""

import os

from dotenv import load_dotenv
load_dotenv()

import pbt

MODELS_DIR = os.path.join(os.path.dirname(__file__), "example_test_run", "models")


# ---------------------------------------------------------------------------
# Example 1 — default: picks up models/rag.py and GEMINI_API_KEY
# ---------------------------------------------------------------------------

def example_default():
    print("=== Example 1: default run ===")
    results = pbt.run(models_dir=MODELS_DIR)
    for name, output in results.items():
        preview = output[:80].replace("\n", " ") if output else "(no output)"
        print(f"  {name:20s}  {preview}")


# ---------------------------------------------------------------------------
# Example 2 — inline llm_call: swap the LLM backend without any files
# ---------------------------------------------------------------------------

def my_llm(prompt: str) -> str:
    """Stub that echoes the prompt length — replace with a real LLM call."""
    return f"[stub] received {len(prompt)} chars"


def example_inline_llm():
    print("\n=== Example 2: inline llm_call ===")
    results = pbt.run(models_dir=MODELS_DIR, llm_call=my_llm)
    for r in results:
        preview = r.llm_output[:80].replace("\n", " ") if r.llm_output else r.error
        print(f"  {r.model_name:20s}  [{r.status}]  {preview}")


# ---------------------------------------------------------------------------
# Example 3 — inline rag_call: override the RAG function at call time
# ---------------------------------------------------------------------------

CUSTOM_ARTICLES = [
    "Getting Started with Prompt Engineering",
    "How to Build a RAG Pipeline from Scratch",
    "LLM Evaluation Frameworks Compared",
]


def my_rag(*args) -> list[str]:
    query = args[0] if args else ""
    query_words = {w.lower() for w in query.split() if len(w) > 3}
    matches = [
        a for a in CUSTOM_ARTICLES
        if query_words & {w.lower() for w in a.split()}
    ]
    return matches or [False]


def example_inline_rag():
    print("\n=== Example 3: inline rag_call + inline llm_call ===")
    results = pbt.run(models_dir=MODELS_DIR, llm_call=my_llm, rag_call=my_rag)
    for r in results:
        preview = r.llm_output[:80].replace("\n", " ") if r.llm_output else r.error
        print(f"  {r.model_name:20s}  [{r.status}]  {preview}")


# ---------------------------------------------------------------------------
# Run all examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if os.environ.get("GEMINI_API_KEY"):
        example_default()
    else:
        print("=== Example 1 skipped (GEMINI_API_KEY not set) ===")

    example_inline_llm()
    example_inline_rag()
