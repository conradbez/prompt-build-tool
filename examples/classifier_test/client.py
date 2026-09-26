"""Classifier judge for this project's tests, served by a local Ollaya.

`pbt test` sends each test's question and text to Ollaya's /v1/systemone
endpoint and passes the test when P(yes) reaches its threshold.
"""

import os

import pbt

classify_call = pbt.systemone_classifier(
    model=os.environ.get("OLLAYA_MODEL", "laya"),
    base_url=os.environ.get("OLLAYA_URL", "http://localhost:11435"),
)
test_judge = "classifier"


def llm_call(prompt: str) -> str:
    # pbt needs an llm_call, but nothing here uses it: the only model is a
    # template and every test is judged by the classifier.
    raise NotImplementedError("This example makes no LLM calls.")
