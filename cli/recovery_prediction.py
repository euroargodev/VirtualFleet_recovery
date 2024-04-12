#!/usr/bin/env python
# -*coding: UTF-8 -*-
#
# This script can be used to make prediction of a specific float cycle position, given:
# - the previous cycle
# - the ARMORD3D or CMEMS GLORYS12 forecast at the time of the previous cycle
#
# mprof run recovery_prediction.py --output data 2903691 80
# mprof plot
# python -m line_profiler recovery_prediction.py --output data 2903691 80
#
# Capital variables are considered global and usable anywhere
#
# Created by gmaze on 06/10/2022
import sys
import os
import glob
import logging
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import requests
import json
from datetime import timedelta
from tqdm import tqdm
import argparse
from argopy.utils import is_wmo, is_cyc, check_cyc
import argopy.plot as argoplot
from argopy.stores.argo_index_pd import indexstore_pandas as store
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import matplotlib
import time
import platform, socket, psutil
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks

import copernicusmarine

logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("parso").setLevel(logging.ERROR)
DEBUGFORMATTER = '%(asctime)s [%(levelname)s] [%(name)s] %(filename)s:%(lineno)d: %(message)s'

log = logging.getLogger("virtualfleet.recovery")


def setup_args():
    icons_help_string = """This script can be used to make prediction of a specific float cycle position.
    This script can be used on past or unknown float cycles.
    Note that in order to download online velocity fields from the Copernicus Marine Data Store, you need to have the 
    appropriate credentials file setup.
        """

    parser = argparse.ArgumentParser(description='VirtualFleet recovery predictor',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="%s\n(c) Argo-France/Ifremer/LOPS, 2022-2024" % icons_help_string)

    # Add long and short arguments
    parser.add_argument('wmo', help="Float WMO number", type=int)
    parser.add_argument("cyc", help="Cycle number to predict", type=int, nargs="+")
    parser.add_argument("--nfloats", help="Number of virtual floats used to make the prediction, default: 2000",
                        type=int, default=2000)
    parser.add_argument("--output", help="Output folder, default: webAPI internal folder", default=None)
    parser.add_argument("--velocity", help="Velocity field to use. Possible values are: 'ARMOR3D' (default), 'GLORYS'",
                        default='ARMOR3D')
    parser.add_argument("--domain_size", help="Size (deg) of the velocity domain to load, default: 12",
                        type=float, default=12)
    parser.add_argument("--save_figure", help="Should we save figure on file or not ? Default: True", default=True)
    parser.add_argument("--save_sim", help="Should we save the simulation on file or not ? Default: False", default=False)
    parser.add_argument("--vf", help="Parent folder to the VirtualFleet repository clone", default=None)
    parser.add_argument("--json", help="Use to only return a json file and stay quiet", action='store_true')

    parser.add_argument("--cfg_parking_depth", help="Virtual floats parking depth in [db], default: use previous cycle value", default=None)
    parser.add_argument("--cfg_cycle_duration", help="Virtual floats cycle duration in [hours], default: use previous cycle value", default=None)
    parser.add_argument("--cfg_profile_depth", help="Virtual floats profiles depth in [db], default: use previous cycle value", default=None)
    parser.add_argument("--cfg_free_surface_drift", help="Virtual floats free surface drift starting cycle number, default: 9999", default=9999)

    return parser




if __name__ == '__main__':
    # Read mandatory arguments from the command line
    ARGS = setup_args().parse_args()

    js = predictor(ARGS)
    if ARGS.json:
        sys.stdout.write(js)

    # Exit gracefully
    sys.exit(0)
