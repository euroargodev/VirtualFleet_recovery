import click
import logging

from vfrecovery.core.predict import predict_function

root_logger = logging.getLogger("vfrecovery_root_logger")
blank_logger = logging.getLogger("vfrecovery_blank_logger")


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
    "-v", "--velocity",
    type=str,
    required=False,
    default='GLORYS',
    show_default=True,
    help="Velocity field to use. Possible values are: 'GLORYS', 'ARMOR3D'",
)
@click.option(
    "--output_path",
    type=str,
    required=False,
    default=None,
    help="Simulation data output folder [default: './vfrecovery_simulations_data/<WMO>/<CYC>']",
)
# @click.option(
#     "-v", "--verbose",
#     type=bool,
#     required=False,
#     is_flag=True,
#     default=True,
#     show_default=True,
#     help="Display verbose information along the execution",
# )
@click.option(
    "--cfg_parking_depth",
    type=float,
    required=False,
    default=None,
    show_default=False,
    help="Virtual floats parking depth in db [default: previous cycle value]",
)
@click.option(
    "--cfg_cycle_duration",
    type=float,
    required=False,
    default=None,
    show_default=False,
    help="Virtual floats cycle duration in hours [default: previous cycle value]",
)
@click.option(
    "--cfg_profile_depth",
    type=float,
    required=False,
    default=None,
    show_default=False,
    help="Virtual floats profile depth in db [default: previous cycle value]",
)
@click.option(
    "--cfg_free_surface_drift",
    type=int,
    required=False,
    default=9999,
    show_default=True,
    help="Virtual cycle number to start free surface drift, inclusive",
)
@click.option(
    "-np", "--n_predictions",
    type=int,
    required=False,
    default=0,
    show_default=True,
    help="Number of profiles to simulate after cycle specified with argument 'CYC'",
)
@click.option(
    "-nf", "--n_floats",
    type=int,
    required=False,
    default=100,
    show_default=True,
    help="Number of virtual floats simulated to make predictions",
)
@click.option(
    "-s", "--domain_min_size",
    type=float,
    required=False,
    default=12,
    show_default=True,
    help="Minimal size (deg) of the simulation domain around the initial float position",
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
@click.argument('CYC', nargs=1, type=int)
def predict(
        wmo,
        cyc,
        velocity,
        output_path,
        n_predictions,
        cfg_parking_depth,
        cfg_cycle_duration,
        cfg_profile_depth,
        cfg_free_surface_drift,
        n_floats,
        domain_min_size,
        log_level,
) -> None:
    if log_level == "QUIET":
        root_logger.disabled = True
        log_level = "CRITICAL"
    root_logger.setLevel(level=getattr(logging, log_level.upper()))

    if root_logger.isEnabledFor(logging.DEBUG):
        root_logger.debug("DEBUG mode activated")

    #
    json_dump = predict_function(wmo, cyc,
                                 velocity=velocity,
                                 output_path=output_path,
                                 n_predictions=n_predictions,
                                 cfg_parking_depth=cfg_parking_depth,
                                 cfg_cycle_duration=cfg_cycle_duration,
                                 cfg_profile_depth=cfg_profile_depth,
                                 cfg_free_surface_drift=cfg_free_surface_drift,
                                 n_floats=n_floats,
                                 domain_min_size=domain_min_size,
                                 log_level=log_level)
    blank_logger.info(json_dump)
