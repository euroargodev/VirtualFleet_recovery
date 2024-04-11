import click
import logging
from argopy.utils import is_wmo, is_cyc, check_cyc, check_wmo
import argopy.plot as argoplot
from argopy import ArgoIndex

from vfrecovery.utils.misc import list_float_simulation_folders
from vfrecovery.core.db import DB

root_logger = logging.getLogger("vfrecovery_root_logger")
blank_logger = logging.getLogger("vfrecovery_blank_logger")


@click.group()
def cli_group_db() -> None:
    pass


@cli_group_db.command(
    "db",
    short_help="Helper for VirtualFleet-Recovery simulations database",
    help="""
    Internal simulation database helper

    """,
    epilog="""
    Examples:

    \b
    vfrecovery db info

    \b
    vfrecovery db read

    \b
    vfrecovery db read --index 3

    \b
    vfrecovery db drop
    """,  # noqa
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL", "QUIET"]),
    default="INFO",
    show_default=True,
    help=(
            "Set the details printed to console by the command "
            "(based on standard logging library)."
    ),
)
@click.option(
    "-i", "--index",
    type=int,
    required=False,
    default=None,
    show_default=False,
    help="Record index to work with",
)
@click.argument('ACTION', nargs=1, type=str)
def db(
        action,
        index,
        log_level,
) -> None:
    if log_level == "QUIET":
        root_logger.disabled = True
        log_level = "CRITICAL"
    root_logger.setLevel(level=getattr(logging, log_level.upper()))

    if root_logger.isEnabledFor(logging.DEBUG):
        root_logger.debug("DEBUG mode activated")

    if action == 'read':
        df = DB.read_data()
        if index is not None:
            row = df.loc[index]
            click.secho("Row index #%i:" % index, fg='green')
            click.echo(row.T.to_string())
        else:
            for irow, row in df.iterrows():
                click.secho("Row index #%i:" % irow, fg='green')
                click.echo(row.T.to_string())

    if action == 'drop':
        DB.clear()

    if action == 'info':
        click.echo(DB.info())
