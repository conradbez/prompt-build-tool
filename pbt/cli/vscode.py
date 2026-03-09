import json
import os


def is_running_in_vscode():
    # TERM_PROGRAM is the most reliable indicator
    return os.environ.get("TERM_PROGRAM") == "vscode"


def setup_vscode_associations():
    vscode_dir = ".vscode"
    settings_file = os.path.join(vscode_dir, "settings.json")

    if not os.path.exists(vscode_dir):
        os.makedirs(vscode_dir)

    settings = {}
    if os.path.exists(settings_file):
        with open(settings_file, "r") as f:
            try:
                settings = json.load(f)
            except json.JSONDecodeError:
                pass

    associations = settings.get("files.associations", {})
    associations["*.prompt"] = "jinja"
    associations["*.prompt.jinja"] = "jinja"
    settings["files.associations"] = associations

    with open(settings_file, "w") as f:
        json.dump(settings, f, indent=4)


