"""
pbt obsidian — use an Obsidian vault's notes as pbt models.

Every subcommand first converts the vault into ``.prompt`` files (see
:mod:`pbt.obsidian.converter`), then hands over to the ordinary pbt command
with ``--models-dir`` pointing at them:

    pbt obsidian build VAULT              # convert only
    pbt obsidian run   VAULT [pbt run options]
    pbt obsidian ls    VAULT
    pbt obsidian test  VAULT [pbt test options]
    pbt obsidian docs  VAULT [pbt docs options]

The generated directory defaults to ``obsidian_models/`` in the current
directory, so ``client.py`` (looked up beside the models directory) is the one
in the current directory, as for a plain ``pbt run``.

``--judge-notes llm|classifier`` asks client.py whether each unmarked note is a
prompt or data (see :mod:`pbt.obsidian.judge`); ``obsidian_judge = "llm"`` in
client.py makes that the default.
"""

from __future__ import annotations

import sys

import click
from rich.markup import escape
from rich.table import Table

from pbt.cli.pretty_print import console, err_console
from pbt.llm import resolve_classify_call, resolve_llm_call, try_load_client_module
from pbt.obsidian import judge as note_judge
from pbt.obsidian.converter import ObsidianError, convert_vault, write_models

DEFAULT_OUT = "obsidian_models"

#: pbt commands that take --models-dir and get an `obsidian` counterpart.
PASSTHROUGH_COMMANDS = ("run", "ls", "test", "docs")

_vault_argument = click.argument("vault", type=click.Path(exists=True, file_okay=False))
_out_option = click.option(
    "--out",
    default=DEFAULT_OUT,
    show_default=True,
    help="Directory the generated .prompt files are written to (replaced on every build).",
)
JUDGE_CHOICES = ("off", "llm", "classifier")
_judge_option = click.option(
    "--judge-notes",
    type=click.Choice(JUDGE_CHOICES),
    default=None,
    help=(
        "Decide prompt vs data for notes without `pbt: prompt|data`, using client.py's "
        "llm_call or classify_call. Data notes become model_type=\"template\". "
        "Default: client.py's obsidian_judge, else off."
    ),
)


def resolve_judge(choice: str | None, out: str) -> note_judge.NoteJudge | None:
    """Build the note judge for *choice*, with client.py looked up beside *out*."""
    try:
        if choice is None:
            module = try_load_client_module(out)
            choice = getattr(module, "obsidian_judge", None) or "off"
            if choice not in JUDGE_CHOICES:
                raise ValueError(f"client.py obsidian_judge must be one of {', '.join(JUDGE_CHOICES)}; got {choice!r}.")
        if choice == "off":
            return None
        if choice == "llm":
            judge = note_judge.llm_judge(resolve_llm_call(out))
        else:
            classify_call = resolve_classify_call(out)
            if classify_call is None:
                raise ValueError("--judge-notes classifier needs a classify_call(state, question) in client.py.")
            judge = note_judge.classifier_judge(classify_call)
    except Exception as exc:
        err_console.print(f"[red]Note judge error:[/red] {escape(str(exc))}")
        sys.exit(1)
    return note_judge.cached(judge, choice)


def build_or_exit(vault: str, out: str, judge_notes: str | None = None, quiet: bool = False) -> None:
    """Convert *vault* into *out*, printing unresolved links; exit on error."""
    judge = resolve_judge(judge_notes, out)
    try:
        converted = convert_vault(vault, judge)
        write_models(converted, out)
    except ObsidianError as exc:
        err_console.print(f"[red]Obsidian error:[/red] {escape(str(exc))}")
        sys.exit(1)

    for note in converted.values():
        for target in note.unresolved:
            err_console.print(
                f"[yellow]Warning:[/yellow] {escape(str(note.note_path))}: {escape(f'[[{target}]]')} is not a note in the vault; kept as text.",
                soft_wrap=True,
            )

    if quiet:
        data = [n.name for n in converted.values() if n.kind == note_judge.DATA]
        data_label = f" ({len(data)} data: {', '.join(data)})" if data else ""
        console.print(
            f"  [dim]Obsidian:[/dim] {len(converted)} notes{escape(data_label)} "
            f"from [cyan]{escape(vault)}[/cyan] → [cyan]{escape(out)}/[/cyan]\n"
        )
        return

    table = Table(title=f"{vault} → {out}/", show_lines=False)
    table.add_column("Note")
    table.add_column("Model", style="cyan")
    table.add_column("Kind")
    table.add_column("Links to")
    for note in converted.values():
        kind = note.kind + (" [dim](judged)[/dim]" if note.kind_source == "judge" else "")
        table.add_row(note.note_path.as_posix(), note.name, kind, ", ".join(note.depends_on) or "[dim]—[/dim]")
    console.print(table)


def register_command(main: click.Group) -> None:
    """Attach the `pbt obsidian` command group to *main*."""

    @main.group("obsidian")
    def obsidian() -> None:
        """Use an Obsidian vault's notes as pbt models ([[links]] become ref())."""

    @obsidian.command("build")
    @_vault_argument
    @_out_option
    @_judge_option
    def build(vault: str, out: str, judge_notes: str | None) -> None:
        """Convert the notes in VAULT into .prompt files."""
        build_or_exit(vault, out, judge_notes)

    for name in PASSTHROUGH_COMMANDS:
        _register_passthrough(main, obsidian, name)


def _register_passthrough(main: click.Group, group: click.Group, name: str) -> None:
    @group.command(
        name,
        help=(
            f"Build VAULT, then `pbt {name} --models-dir OUT`. "
            f"Options after VAULT that this command does not know go to `pbt {name}`."
        ),
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )
    @_vault_argument
    @_out_option
    @_judge_option
    @click.pass_context
    def passthrough(ctx: click.Context, vault: str, out: str, judge_notes: str | None) -> None:
        build_or_exit(vault, out, judge_notes, quiet=True)
        target = main.commands[name]
        with target.make_context(name, ["--models-dir", out, *ctx.args], parent=ctx) as sub:
            target.invoke(sub)
