# Quality model node type

The quality check will be used the inject_nodes_before hook to expand a single node into a chain of nodes at DAG-build time, before execution begins.
Given a DAG like:
article → article_quality → summary

When apply_replace_node_callbacks ran, it saw article_quality had model_type="quality_check" and called inject_nodes_before(article_quality, all_models). That function returned a list of replacement nodes:
article_quality_1   (intermediate check — did it pass?)
article_1           (retry — skips if quality_1 passed)
article_quality_2   (intermediate check)
article_2           (retry — skips if quality_2 passed)
article_quality     (terminal pass-through — outputs the best article)

The original article_quality node was deleted from the DAG and these five nodes were inserted in its place. The terminal node kept the original name (article_quality) so summary's ref('article_quality') still resolved correctly — no downstream nodes needed rewriting.
So the pattern was: one node in, N nodes out. The handler owned the entire sub-chain and the rest of the DAG was unaware it had been expanded. The executor then just ran the expanded DAG normally — no special quality-check logic needed at execution time.