import time
from argopy.utils import is_wmo, is_cyc, check_cyc, check_wmo
from pathlib import Path
from typing import Union
import os
import logging
import json
import pandas as pd

from .simulation_handler import Simulation
from .utils import pp_obj, get_a_log_filename

root_logger = logging.getLogger("vfrecovery_root_logger")
sim_logger = logging.getLogger("vfrecovery_simulation")


class log_this:

    def __init__(self, txt, log_level):
        """Log text to simulation and possibly root logger(s)"""
        getattr(root_logger, log_level.lower())(txt)
        getattr(sim_logger, log_level.lower())(txt)

    @staticmethod
    def info(txt) -> 'log_this':
        return log_this(txt, 'INFO')

    @staticmethod
    def debug(txt) -> 'log_this':
        return log_this(txt, 'DEBUG')

    @staticmethod
    def warning(txt) -> 'log_this':
        return log_this(txt, 'WARNING')

    @staticmethod
    def error(txt) -> 'log_this':
        return log_this(txt, 'ERROR')


def predict_function(
        wmo: int,
        cyc: int,
        velocity: str,
        n_predictions: int,
        output_path: Union[str, Path],
        cfg_parking_depth: float,
        cfg_cycle_duration: float,
        cfg_profile_depth: float,
        cfg_free_surface_drift: int,
        swarm_size: int,
        domain_min_size: float,
        overwrite: bool,
        lazy: bool,
        figure: bool,
        log_level: str,
) -> str:
    """
    Execute VirtualFleet-Recovery predictor and save results as a JSON string

    Parameters
    ----------
    wmo
    cyc
    velocity
    n_predictions
    output_path
    cfg_parking_depth
    cfg_cycle_duration
    cfg_profile_depth
    cfg_free_surface_drift
    swarm_size
    domain_min_size
    overwrite
    lazy
    figure
    log_level

    Returns
    -------
    str: a JSON formatted str

    """  # noqa
    if log_level == "QUIET":
        root_logger.disabled = True
        log_level = "CRITICAL"
    root_logger.setLevel(level=getattr(logging, log_level.upper()))

    execution_start = time.time()
    process_start = time.process_time()
    # run_id = pd.to_datetime('now', utc=True).strftime('%Y%m%d%H%M%S')

    # Validate arguments:
    assert is_wmo(wmo)
    assert is_cyc(cyc)
    wmo = check_wmo(wmo)[0]
    cyc = check_cyc(cyc)[0]

    if velocity.upper() not in ['ARMOR3D', 'GLORYS']:
        raise ValueError("Velocity field must be one in: ['ARMOR3D', 'GLORYS']")
    else:
        velocity = velocity.upper()

    # Build the list of cycle numbers to work with `cyc`:
    # The `cyc` list follows this structure:
    #   [PREVIOUS_CYCLE_USED_AS_INITIAL_CONDITIONS, CYCLE_NUMBER_REQUESTED_BY_USER, ADDITIONAL_CYCLE_i, ADDITIONAL_CYCLE_i+1, ...]
    # Prepend previous cycle number that will be used as initial conditions for the prediction of `cyc`:
    cyc = [cyc - 1, cyc]
    # Append additional `n_predictions` cycle numbers:
    [cyc.append(cyc[1] + n + 1) for n in range(n_predictions)]

    if output_path is None:
        output_path = Path(__file__).parents[2].joinpath("vfrecovery_simulations_data")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set-up simulation logger
    templogfile = get_a_log_filename(output_path, name='simulation_')
    simlogfile = logging.FileHandler(templogfile, mode='a')
    simlogfile.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s:%(filename)s | %(message)s",
                                              datefmt='%Y/%m/%d %I:%M:%S'))
    sim_logger.handlers = []
    sim_logger.addHandler(simlogfile)

    # Redirect all warnings to log files
    logging.captureWarnings(True)

    #
    S = Simulation(wmo, cyc,
                   swarm_size=swarm_size,
                   velocity=velocity,
                   output_path=output_path,
                   overwrite=overwrite,
                   lazy=lazy,
                   logger=log_this,
                   figure=figure,
                   )
    S.setup(cfg_parking_depth=cfg_parking_depth,
            cfg_cycle_duration=cfg_cycle_duration,
            cfg_profile_depth=cfg_profile_depth,
            cfg_free_surface_drift=cfg_free_surface_drift,
            domain_min_size=domain_min_size,
            )
    if not S.is_registered or overwrite:
        S.execute()
        S.predict()
        S.postprocess()
        S.finish(execution_start, process_start)  # Save on disk in json file
    else:
        log_this.info("This simulation already exists, stop here and return existing results")

    # Move log file to the appropriate final destination:
    templogfile.rename(get_a_log_filename(S.output_path))

    # Load json results to return
    with open(S.run_file, 'r') as f:
        jsdata = json.load(f)
    return json.dumps(jsdata, indent=4)



# def predictor(args):
#     """Prediction manager"""
#
#     if args.save_figure:
#         mplbackend = matplotlib.get_backend()
#         matplotlib.use('Agg')

#     if args.save_figure:
#         plt.close('all')
#         # Restore Matplotlib backend
#         matplotlib.use(mplbackend)
#
#     if not args.save_sim:
#         shutil.rmtree(output_path)
#
#     return results_js
#
