import click
from typing import Union, List
from vfrecovery.core.predict import predict_function

@click.group()
def cli_group_predict() -> None:
    pass

@cli_group_predict.command(
    "predict",
    short_help="Execute VirtualFleet-Recovery predictions",
    help="""
    Execute VirtualFleet-Recovery predictor and return results as a JSON string
    """,
    epilog="""
    Examples:

    \b
    vfrecovery predict 6903091 112
    """,  # noqa
 )
@click.option(
    "-n", "--n_predictions",
    type=int,
    required=False,
    default=1,
    show_default=True,
    is_flag=False,
    help="Number of profiles to simulate",
)
@click.argument('WMO')
@click.argument('CYC', nargs=-1)
def predict(
        wmo: int,
        cyc: Union[int, List],
        n_predictions) -> None:
    # click.echo(f"Prediction for {wmo} {cyc}")
    results = predict_function(wmo, cyc, n_predictions=n_predictions)
    click.echo(results)
