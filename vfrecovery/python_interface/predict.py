import json
from vfrecovery.core.predict import predict_function
from pathlib import Path
from typing import Union


def predict(
        wmo: int,
        cyc: int,
        velocity: str = 'GLORYS',
        output_path: Union[str, Path] = None,
        n_predictions: int = 0,
        cfg_parking_depth: float = None,
        cfg_cycle_duration: float = None,
        cfg_profile_depth: float = None,
        cfg_free_surface_drift: int = 9999,
        swarm_size: int = 100,
        domain_min_size: float = 5.,
        overwrite: bool = False,
        lazy: bool = True,
        figure: bool = True,
        log_level: str = 'INFO',
):
    """
    Execute VirtualFleet-Recovery predictor and return results as a JSON string
    
    Parameters
    ----------
    wmo
    cyc
    velocity
    output_path
    n_predictions
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
    data
        
    """  # noqa
    results_json = predict_function(
        wmo, cyc,
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
        log_level=log_level,
    )
    results = json.loads(results_json)
    return results
