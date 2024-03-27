import argopy.plot as argoplot
from pathlib import Path
import os
from argopy.utils import is_cyc


def get_package_dir():
    fpath = Path(__file__)
    return str(fpath.parent.parent)


def get_cfg_str(a_cfg):
    txt = "VFloat configuration: (Parking depth: %i [db], Cycle duration: %i [hours], Profile depth: %i [db])" % (
        a_cfg.mission['parking_depth'],
        a_cfg.mission['cycle_duration'],
        a_cfg.mission['profile_depth'],
    )
    return txt


def list_float_simulation_folders(wmo, cyc=None) -> dict:
    """Return the list of all available simulation folders for a given WMO"""
    output_path = {}
    pl = Path(os.path.sep.join(["vfrecovery_simulations_data", str(wmo)])).glob("*")
    for p in pl:
        if p.is_dir():
            cyc = p.parts[-1]
            if is_cyc(cyc):
                output_path.update({int(cyc): p})
    if cyc is not None:
        output_path = {c: output_path[c] for c in list(output_path) if str(c) in cyc}
    return output_path


def get_ea_profile_page_url(wmo, cyc):
    try:
        url = argoplot.dashboard(wmo, cyc, url_only=True)
    except:
        # log.info("EA dashboard page not available for this profile: %i/%i" % (wmo, cyc))
        url = "404"
    return url
