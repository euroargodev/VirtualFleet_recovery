import click
import logging
from argopy.utils import is_wmo, is_cyc, check_cyc, check_wmo
import argopy.plot as argoplot
from argopy.errors import DataNotFound
from argopy import ArgoIndex


from vfrecovery.core.describe import describe_function

root_logger = logging.getLogger("vfrecovery_root_logger")
blank_logger = logging.getLogger("vfrecovery_blank_logger")



@click.group()
def cli_group_describe() -> None:
    pass

@cli_group_describe.command(
    "describe",
    short_help="Describe VirtualFleet-Recovery simulation results",
    help="""
    Returns data about an existing VirtualFleet-Recovery prediction
    
    Data could be a JSON file, specific metrics or images
    """,
    epilog="""
    Examples:

    \b
    vfrecovery describe 6903091

    \b
    vfrecovery describe 6903091 112
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
@click.argument('WMO', nargs=1, type=int)
@click.argument("CYC", nargs=-1, type=int)
def describe(
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
    assert is_wmo(wmo)
    wmo = check_wmo(wmo)[0]
    cyc = list(cyc)
    if len(cyc) > 0:
        assert is_cyc(cyc)
        cyc = check_cyc(cyc)

    #
    # json_dump = describe_function(wmo,
    #                               cyc=cyc,
    #                              log_level=log_level)
    # blank_logger.info(json_dump)

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
