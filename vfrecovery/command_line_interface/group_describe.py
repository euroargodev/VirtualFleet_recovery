import click


@click.group()
def cli_group_describe() -> None:
    pass

@cli_group_describe.command(
    "describe",
    short_help="Describe VirtualFleet-Recovery predictions",
    help="""
    Returns data about an existing VirtualFleet-Recovery prediction
    
    Data could be a JSON file, specific metrics or images
    """,
    epilog="""
    Examples:

    \b
    vfrecovery describe 6903091 112
    """,  # noqa
 )
@click.argument('WMO')
@click.argument('CYC')
def describe(
        wmo: int,
        cyc: int):
    click.echo(f"Return description for {wmo} {cyc}")