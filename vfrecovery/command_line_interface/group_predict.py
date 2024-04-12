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
    Execute the VirtualFleet-Recovery predictor
    
    WMO is the float World Meteorological Organisation number.
    
    CYC is the cycle number to predict. If you want to simulate more than 1 cycle, use the `n_predictions` option (see below).
    """,
    epilog="""
Examples:
\b
\n\tvfrecovery predict 6903091 112
    """,  # noqa
)
@click.option(
    "-v", "--velocity",
    type=str,
    required=False,
    default='GLORYS',
    show_default=True,
    help="Velocity field to use. Velocity data are downloaded with the Copernicus Marine Toolbox. Possible values are: 'GLORYS', 'ARMOR3D'",
)
@click.option(
    "--output_path",
    type=str,
    required=False,
    default=None,
    help="Simulation root data output folder [default: './vfrecovery_simulations_data']",
)
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
    "-n", "--n_predictions",
    type=int,
    required=False,
    default=0,
    show_default=True,
    help="Number of profiles to predict after cycle specified with argument 'CYC'",
)
@click.option(
    "-s", "--swarm_size",
    type=int,
    required=False,
    default=100,
    show_default=True,
    help="Swarm size, i.e. the number of virtual floats simulated to make predictions for 1 real float",
)
@click.option(
    "-d", "--domain_min_size",
    type=float,
    required=False,
    default=5,
    show_default=True,
    help="Minimal size (deg) of the simulation domain around the initial float position",
)
@click.option('--overwrite',
              is_flag=True,
              help="Should past simulation data be overwritten or not, for a similar set of arguments"
              )
@click.option('--lazy/--no-lazy',
              default=True,
              show_default=True,
              help="Load velocity data in lazy mode (not saved on file)."
              )
@click.option('--figure/--no-figure',
              default=True,
              show_default=True,
              help="Display and save figures on file (png format)",
              )
@click.option(
    "--log_level",
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
        swarm_size,
        domain_min_size,
        overwrite,
        lazy,
        figure,
        log_level,
) -> None:
    """

    """
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
                                 swarm_size=swarm_size,
                                 domain_min_size=domain_min_size,
                                 overwrite=overwrite,
                                 lazy=lazy,
                                 figure=figure,
                                 log_level=log_level)
    blank_logger.info(json_dump)
