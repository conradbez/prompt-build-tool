# pbt-extra-model-kinds

Extra model kinds for [pbt](https://pypi.org/project/prompt-build-tool/):

| `model_type`   | Module          | Runs                                                   |
|----------------|-----------------|--------------------------------------------------------|
| `python_modal` | `modal_exec`    | the bullet's upstream Python in a Modal sandbox        |
| `agent_modal`  | `agent_exec`    | an OpenCode agent (plus optional MCP server) on Modal  |
| `agent37`      | `agent37_exec`  | a Hermes agent on an Agent37 workspace (`agent37.md`)  |

Installing the package is enough: pbt >= 0.5.10 finds the kinds through the
`pbt.model_kinds` entry point group and loads them on first use. Nothing has
to import it.

Configuration is read from the environment — see each module's docstring.
