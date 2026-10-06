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
"""

from __future__ import annotations

import sys

import click
from rich.markup import escape
from rich.table import Table

from pbt.cli.pretty_print import console, err_console
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


def build_or_exit(vault: str, out: str, quiet: bool = False) -> None:
    """Convert *vault* into *out*, printing unresolved links; exit on error."""
    try:
        converted = convert_vault(vault)
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
        console.print(f"  [dim]Obsidian:[/dim] {len(converted)} notes from [cyan]{vault}[/cyan] → [cyan]{out}/[/cyan]\n")
        return

    table = Table(title=f"{vault} → {out}/", show_lines=False)
    table.add_column("Note")
    table.add_column("Model", style="cyan")
    table.add_column("Links to")
    for note in converted.values():
        table.add_row(note.note_path.as_posix(), note.name, ", ".join(note.depends_on) or "[dim]—[/dim]")
    console.print(table)


def register_command(main: click.Group) -> None:
    """Attach the `pbt obsidian` command group to *main*."""

    @main.group("obsidian")
    def obsidian() -> None:
        """Use an Obsidian vault's notes as pbt models ([[links]] become ref())."""

    @obsidian.command("build")
    @_vault_argument
    @_out_option
    def build(vault: str, out: str) -> None:
        """Convert the notes in VAULT into .prompt files."""
        build_or_exit(vault, out)

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
    @click.pass_context
    def passthrough(ctx: click.Context, vault: str, out: str) -> None:
        build_or_exit(vault, out, quiet=True)
        target = main.commands[name]
        with target.make_context(name, ["--models-dir", out, *ctx.args], parent=ctx) as sub:
            target.invoke(sub)
