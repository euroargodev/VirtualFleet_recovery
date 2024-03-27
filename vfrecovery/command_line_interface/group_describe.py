import click
import logging
from argopy.utils import is_wmo, is_cyc, check_cyc, check_wmo
import argopy.plot as argoplot
from argopy import ArgoIndex

from vfrecovery.utils.misc import list_float_simulation_folders


root_logger = logging.getLogger("vfrecovery_root_logger")
blank_logger = logging.getLogger("vfrecovery_blank_logger")


@click.group()
def cli_group_describe() -> None:
    pass


@cli_group_describe.command(
    "describe",
    short_help="Describe VirtualFleet-Recovery data and simulation results",
    help="""

    TARGET select what is to be described. A string in: ['obs', 'velocity'].
    
    WMO is the float World Meteorological Organisation number
    
    CYC is the cycle number location to restrict description to
    """,
    epilog="""
    Examples:

    \b
    vfrecovery describe velocity 6903091

    \b
    vfrecovery describe obs 6903091 112
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
def describe(
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

    if target == 'obs':
        describe_obs(wmo, cyc)

    elif target == 'velocity':
        describe_velocity(wmo, cyc)


def describe_velocity(wmo, cyc):

    # List folders to examine:
    plist = list_float_simulation_folders(wmo, cyc)

    # List all available velocity files:
    for c in plist.keys():
        p = plist[c]
        click.secho("Velocity data for WMO=%s / CYC=%s:" % (wmo, c), fg='blue')

        click.secho("\tNetcdf files:")
        vlist = sorted(p.glob("velocity_*.nc"))
        if len(vlist) > 0:
            [click.secho("\t\t- %s" % v, fg='green') for v in vlist]
        else:
            click.secho("\tNo velocity file", fg='red')

        click.secho("\tFigures:")
        vlist = sorted(p.glob("velocity_*.png"))
        if len(vlist) > 0:
            [click.secho("\t\t- %s" % v, fg='green') for v in vlist]
        else:
            click.secho("\tNo velocity figures", fg='red')


def describe_obs(wmo, cyc):
    url = argoplot.dashboard(wmo, url_only=True)
    # txt = "You can check this float dashboard while we search for float profiles in the index: %s" % url
    click.secho("\nYou can check this float dashboard while we search for float profile(s) in the index:")
    click.secho("\t%s" % url, fg='green')

    # Load observed float profiles index:
    host = "https://data-argo.ifremer.fr"
    # host = "/home/ref-argo/gdac" if os.uname()[0] == 'Darwin' else "https://data-argo.ifremer.fr"
    # host = "/home/ref-argo/gdac" if not os.uname()[0] == 'Darwin' else "~/data/ARGO"
    idx = ArgoIndex(host=host)
    if len(cyc) > 0:
        idx.search_wmo_cyc(wmo, cyc)
    else:
        idx.search_wmo(wmo)

    df = idx.to_dataframe()
    df = df.sort_values(by='date').reset_index(drop=True)
    if df.shape[0] == 1:
        click.secho("\nFloat profile data from the index:")
        # df_str = "\t%s" % (df.T).to_string()
        df_str = "\n".join(["\t%s" % l for l in (df.T).to_string().split("\n")[1:]])
        click.secho(df_str, fg="green")
    else:
        click.secho("\nFloat profile(s):")
        # click.secho(df.to_string(max_colwidth=15), fg="green")
        click.secho(df.to_string(), fg="green")
    # click.echo_via_pager("\n%s" % df.to_string(max_colwidth=15))
