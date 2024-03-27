from virtualargofleet import FloatConfiguration, ConfigParam


def setup_floats_config(
        wmo: int,
        cyc: int,
        cfg_parking_depth: float,
        cfg_cycle_duration: float,
        cfg_profile_depth: float,
        cfg_free_surface_drift: int,
        logger,
) -> FloatConfiguration:
    """Load float configuration at a given cycle number and possibly overwrite data with user parameters

    Parameters
    ----------
    wmo: int
    cyc: int
    cfg_parking_depth: float,
    cfg_cycle_duration: float,
    cfg_profile_depth: float,
    cfg_free_surface_drift: int,
    logger

    Returns
    -------
    :class:`virtualargofleet.FloatConfiguration`

    """
    try:
        CFG = FloatConfiguration([wmo, cyc])
    except:
        logger.error("Can't load this profile configuration, fall back on default values")
        CFG = FloatConfiguration('default')

    if cfg_parking_depth is not None:
        logger.debug("parking_depth=%i is overwritten with %i" % (CFG.mission['parking_depth'],
                                                                    float(cfg_parking_depth)))
        CFG.update('parking_depth', float(cfg_parking_depth))

    if cfg_cycle_duration is not None:
        logger.debug("cycle_duration=%i is overwritten with %i" % (CFG.mission['cycle_duration'],
                                                                     float(cfg_cycle_duration)))
        CFG.update('cycle_duration', float(cfg_cycle_duration))

    if cfg_profile_depth is not None:
        logger.debug("profile_depth=%i is overwritten with %i" % (CFG.mission['profile_depth'],
                                                                    float(cfg_profile_depth)))
        CFG.update('profile_depth', float(cfg_profile_depth))

    CFG.params = ConfigParam(key='reco_free_surface_drift',
                             value=int(cfg_free_surface_drift),
                             unit='cycle',
                             description='First cycle with free surface drift',
                             dtype=int)

    return CFG

