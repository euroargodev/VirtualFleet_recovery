import argopy.plot as argoplot
from pathlib import Path


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




def get_ea_profile_page_url(wmo, cyc):
    try:
        url = argoplot.dashboard(wmo, cyc, url_only=True)
    except:
        # log.info("EA dashboard page not available for this profile: %i/%i" % (wmo, cyc))
        url = "404"
    return url
