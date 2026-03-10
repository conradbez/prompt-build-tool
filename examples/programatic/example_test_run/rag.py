ARTICLES = [
    "How Large Language Models Are Changing Software Development",
    "A Beginner's Guide to Open-Source AI Frameworks",
    "The Rise of Vector Databases in Modern Applications",
    "Understanding Retrieval-Augmented Generation",
    "Developer Tooling in the Age of AI Assistants",
    "Fine-Tuning vs Prompting: When to Use Each Approach",
    "Building Production-Ready LLM Pipelines",
    "Open-Source Models vs Proprietary APIs: A Practical Comparison",
    "How to Evaluate Your LLM Application",
    "The Future of AI-Powered Developer Workflows",
]


def do_RAG(query: str):
    query_words = {w.lower().strip(".,!?") for w in query.split() if len(w) > 3}
    for article in ARTICLES:
        article_words = {w.lower().strip(".,!?") for w in article.split()}
        if query_words & article_words:
            return article
    return False
