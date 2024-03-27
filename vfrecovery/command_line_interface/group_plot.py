import click
import logging
from argopy.utils import is_wmo, is_cyc, check_cyc, check_wmo
import argopy.plot as argoplot
from argopy.errors import DataNotFound
from argopy import ArgoIndex
import os
from pathlib import Path
import glob

from vfrecovery.core.plot import plot_velocity

root_logger = logging.getLogger("vfrecovery_root_logger")
blank_logger = logging.getLogger("vfrecovery_blank_logger")


@click.group()
def cli_group_plot() -> None:
    pass


@cli_group_plot.command(
    "plot",
    short_help="Plot VirtualFleet-Recovery data or simulation results",
    help="""

    TARGET select what is to be plotted. A string in: 'velocity'.

    WMO is the float World Meteorological Organisation number

    CYC is the cycle number location to restrict plots to
    """,
    epilog="""
    Examples:

    \b
    vfrecovery plot velocity 6903091 80
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
@click.argument('TARGET', nargs=1, type=str)
@click.argument('WMO', nargs=1, type=int)
@click.argument("CYC", nargs=-1, type=int)
def plot(
        target,
        wmo,
        cyc,
        log_level,
) -> None:
    if log_level == "QUIET":
        root_logger.disabled = True
        log_level = "CRITICAL"
    root_logger.setLevel(level=getattr(logging, log_level.upper()))

    if root_logger.isEnabledFor(logging.DEBUG):
        root_logger.debug("DEBUG mode activated")

    # Validate arguments:
    if target.lower() not in ["all", "obs", "velocity"]:
        raise ValueError("The first argument TARGET must be one in ['all', 'obs', 'velocity']")

    assert is_wmo(wmo)
    wmo = check_wmo(wmo)[0]
    cyc = list(cyc)
    if len(cyc) > 0:
        assert is_cyc(cyc)
        cyc = check_cyc(cyc)

    if target == 'velocity':
        plot_velocity(wmo, cyc,
                      log_level=log_level,
                      )
