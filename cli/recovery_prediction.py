#!/usr/bin/env python
# -*coding: UTF-8 -*-
#
# This script can be used to make prediction of a specific float cycle position, given:
# - the previous cycle
# - the ARMORD3D or CMEMS GLORYS12 forecast at the time of the previous cycle
#
#
# mprof run ../cli/recovery_prediction.py --output data 2903691 80
# mprof plot
# python -m line_profiler ../cli/recovery_prediction.py --output data 2903691 80
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
from argopy.utilities import is_wmo, is_cyc
import argopy.plot as argoplot
from argopy.stores.argo_index_pd import indexstore_pandas as store
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import matplotlib
# from parcels import ParticleSet, FieldSet, Field
# from abc import ABC
import time
# from memory_profiler import profile
import platform, socket, psutil
from sklearn.metrics import pairwise_distances

logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("parso").setLevel(logging.ERROR)
DEBUGFORMATTER = '%(asctime)s [%(levelname)s] [%(name)s] %(filename)s:%(lineno)d: %(message)s'

log = logging.getLogger("virtualfleet.recovery")

PREF = "\033["
RESET = f"{PREF}0m"
class COLORS:
    black = "30m"
    red = "31m"
    green = "32m"
    yellow = "33m"
    blue = "34m"
    magenta = "35m"
    cyan = "36m"
    white = "37m"


def get_package_dir():
    fpath = Path(__file__)
    return str(fpath.parent.parent)


def puts(text, color=None, bold=False, file=sys.stdout):
    """Alternative to print, uses no color by default but accepts any color from the COLORS class.

    Parameters
    ----------
    text
    color
    bold
    file
    """
    if color is None:
        txt = f'{PREF}{1 if bold else 0}m' + text + RESET
        print(txt, file=file)
    else:
        txt = f'{PREF}{1 if bold else 0};{color}' + text + RESET
        print(txt, file=file)
    log.info(text)


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    see: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points

    Parameters
    ----------
    lon1
    lat1
    lon2
    lat2
    """
    from math import radians, cos, sin, asin, sqrt
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers.
    return c * r


def bearing(lon1, lat1, lon2, lat2):
    """

    Parameters
    ----------
    lon1
    lat1
    lon2
    lat2

    Returns
    -------

    """
    # from math import cos, sin, atan2, degrees
    # b = atan2(cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1), sin(lon2 - lon1) * cos(lat2))
    # b = degrees(b)
    # return b

    import pyproj
    geodesic = pyproj.Geod(ellps='WGS84')
    fwd_azimuth, back_azimuth, distance = geodesic.inv(lon1, lat1, lon2, lat2)
    return fwd_azimuth


def strfdelta(tdelta, fmt):
    """

    Parameters
    ----------
    tdelta
    fmt

    Returns
    -------

    """
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


def fixLON(x):
    """Ensure a 0-360 longitude"""
    if x < 0:
        x = 360 + x
    return x


def getSystemInfo():
    """Return system information as a dict"""
    try:
        info = {}
        info['platform']=platform.system()
        info['platform-release']=platform.release()
        info['platform-version']=platform.version()
        info['architecture']=platform.machine()
        info['hostname']=socket.gethostname()
        info['ip-address']=socket.gethostbyname(socket.gethostname())
        # info['mac-address']=':'.join(re.findall('..', '%012x' % uuid.getnode()))
        info['processor']=platform.processor()
        info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
        return info
    except Exception as e:
        logging.exception(e)


def get_glorys_forecast_with_opendap(a_box, a_start_date, n_days=1):
    """Load Global Ocean 1/12° Physics Analysis and Forecast updated Daily

    Fields: 6-hourly, from 2020-01-01T00:00 to 'now' + 2 days
    Src: https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSIS_FORECAST_PHY_001_024

    Parameters
    ----------
    a_box
    a_start_date
    n_days

    Returns
    -------
    :class:xarray.dataset
    """
    MOTU_USERNAME, MOTU_PASSWORD = (
        os.getenv("MOTU_USERNAME"),
        os.getenv("MOTU_PASSWORD"),
    )
    if not MOTU_USERNAME:
        raise ValueError("No MOTU_USERNAME in environment ! ")

    session = requests.Session()
    session.auth = (MOTU_USERNAME, MOTU_PASSWORD)
    # 6-hourly fields:
    serverset = 'https://nrt.cmems-du.eu/thredds/dodsC/global-analysis-forecast-phy-001-024-3dinst-uovo'
    # Daily fields:
    # serverset = 'https://nrt.cmems-du.eu/thredds/dodsC/global-analysis-forecast-phy-001-024'
    store = xr.backends.PydapDataStore.open(serverset, session=session)
    ds = xr.open_dataset(store)
    # puts(ds.__repr__())
    # puts("\t%s" % serverset, color=COLORS.green)

    # Get the starting date:
    t = "%0.4d-%0.2d-%0.2d %0.2d:00:00" % (a_start_date.year, a_start_date.month, a_start_date.day,
                                           np.array([0, 6, 12, 18])[
                                               np.argwhere(np.array([0, 6, 12, 18]) + 6 > a_start_date.hour)[0][0]])
    t = np.datetime64(pd.to_datetime(t))

    if t < ds['time'][0]:
        raise ValueError("This float cycle is too old for this velocity field.\n%s < %s" % (t, ds['time'][0].values))

    nt = n_days * 4  # 4 snapshot a day (6 hourly), over n_days days
    itim = np.argwhere(ds['time'].values >= t)[0][0], np.argwhere(ds['time'].values >= t)[0][0] + nt
    if itim[1] > len(ds['time']):
        puts("Requested time frame out of max range (%s). Fall back on the longest time frame available." %
                      pd.to_datetime(ds['time'][-1].values).strftime("%Y-%m-%dT%H:%M:%S"), color=COLORS.yellow)
        itim = np.argwhere(ds['time'].values >= t)[0][0], len(ds['time'])

    idpt = np.argwhere(ds['depth'].values > 2000)[0][0]
    ilon = np.argwhere(ds['longitude'].values >= a_box[0])[0][0], np.argwhere(ds['longitude'].values >= a_box[1])[0][0]
    ilat = np.argwhere(ds['latitude'].values >= a_box[2])[0][0], np.argwhere(ds['latitude'].values >= a_box[3])[0][0]
    glorys = ds.isel({'time': range(itim[0], itim[1]),
                      'depth': range(0, idpt),
                      'longitude': range(ilon[0], ilon[1]),
                      'latitude': range(ilat[0], ilat[1])})

    #
    return glorys.load()


def get_glorys_reanalysis_with_opendap(a_box, a_start_date, n_days=1):
    """Load GLORYS Re-analysis

    Fields: daily, from 1993-01-01T12:00 to 2020-05-31T12:00
    Src: https://resources.marine.copernicus.eu/product-detail/GLOBAL_MULTIYEAR_PHY_001_030

    Parameters
    ----------
    a_box
    a_start_date
    n_days
    """
    MOTU_USERNAME, MOTU_PASSWORD = (
        os.getenv("MOTU_USERNAME"),
        os.getenv("MOTU_PASSWORD"),
    )
    if not MOTU_USERNAME:
        raise ValueError("No MOTU_USERNAME in environment ! ")

    session = requests.Session()
    session.auth = (MOTU_USERNAME, MOTU_PASSWORD)
    # Daily from 1993-01-01 to 2020-05-31:
    serverset = 'https://my.cmems-du.eu/thredds/dodsC/cmems_mod_glo_phy_my_0.083_P1D-m'
    store = xr.backends.PydapDataStore.open(serverset, session=session)
    ds = xr.open_dataset(store)
    # puts(ds.__repr__())
    # puts("\t%s" % serverset, color=COLORS.green)

    if a_start_date > ds['time'][-1]:
        raise ValueError("This float cycle is too young for this velocity field.\n%s > %s" % (a_start_date, ds['time'][-1].values))

    itim = np.argwhere(ds['time'].values<a_start_date)[-1][0], np.argwhere(ds['time'].values<a_start_date+(n_days+1)*pd.Timedelta(1,'D'))[-1][0]+1
    if itim[1] > len(ds['time']):
        print("Requested time frame out of max range (%s). Fall back on the longest time frame available." %
                      pd.to_datetime(ds['time'][-1].values).strftime("%Y-%m-%dT%H:%M:%S"))
        itim = np.argwhere(ds['time'].values < a_start_date)[-1][0], len(ds['time'])
    idpt = np.argwhere(ds['depth'].values>2000)[0][0]
    ilon = np.argwhere(ds['longitude'].values>=a_box[0])[0][0], np.argwhere(ds['longitude'].values>=a_box[1])[0][0]
    ilat = np.argwhere(ds['latitude'].values>=a_box[2])[0][0], np.argwhere(ds['latitude'].values>=a_box[3])[0][0]
    glorys = ds.isel({'time': range(itim[0], itim[1]),
                      'depth': range(0, idpt),
                      'longitude': range(ilon[0], ilon[1]),
                      'latitude': range(ilat[0], ilat[1])})
    #
    return glorys.load()


def get_glorys_with_opendap(a_box, a_start_date, n_days=1):
    """Load Global Ocean 1/12° Physics Re-Analysis and Forecast updated Daily

    If ``a_start_date+n_days`` < 2020-05-31:
        delivers the multi-year reprocessed (REP) daily data
        https://resources.marine.copernicus.eu/product-detail/GLOBAL_MULTIYEAR_PHY_001_030

    otherwise:
        delivers near-real-time (NRT) Analysis and Forecast 6-hourly data
        https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSIS_FORECAST_PHY_001_024

    Parameters
    ----------
    a_box
    a_start_date
    n_days

    Returns
    -------
    :class:xarray.dataset
    """
    if a_start_date + pd.Timedelta(n_days, 'D') <= pd.to_datetime('2020-05-31'):
        loader = get_glorys_reanalysis_with_opendap
    else:
        loader = get_glorys_forecast_with_opendap

    return loader(a_box, a_start_date, n_days=n_days)


def get_glorys_forecast_from_datarmor(a_box, a_start_date, n_days=1):
    """Load Datarmor Global Ocean 1/12° Physics Analysis and Forecast updated Daily

    Fields: daily, from 2020-11-25T12:00 to 'now' + 5 days
    Src: /home/ref-ocean-model-public/multiparameter/physic/global/cmems/global-analysis-forecast-phy-001-024
    Info: https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSIS_FORECAST_PHY_001_024/INFORMATION

    Parameters
    ----------
    a_box
    a_start_date
    n_days
    """
    def get_forecast_files(a_date, n_days=1):
        file_list = []
        for n in range(0, n_days):
            t = a_date + pd.Timedelta(n, 'D')
            p = os.path.join(src, "%i" % t.year, "%0.3d" % t.day_of_year)
            # print(p, os.path.exists(p))
            if os.path.exists(p):
                file_list.append(sorted(glob.glob(os.path.join(p, "*.nc")))[0])
        return file_list

    def preprocess(this_ds):
        idpt = np.argwhere(this_ds['depth'].values > 2000)[0][0]
        ilon = np.argwhere(this_ds['longitude'].values >= a_box[0])[0][0], \
               np.argwhere(this_ds['longitude'].values >= a_box[1])[0][0]
        ilat = np.argwhere(this_ds['latitude'].values >= a_box[2])[0][0], \
               np.argwhere(this_ds['latitude'].values >= a_box[3])[0][0]
        this_ds = this_ds.isel({'depth': range(0, idpt),
                                'longitude': range(ilon[0], ilon[1]),
                                'latitude': range(ilat[0], ilat[1])})
        return this_ds

    root = "/home/ref-ocean-model-public" if not os.uname()[0] == 'Darwin' else "/Volumes/MODEL-PUBLIC/"
    src = os.path.join(root, "multiparameter/physic/global/cmems/global-analysis-forecast-phy-001-024")
    # puts("\t%s" % src, color=COLORS.green)
    flist = get_forecast_files(a_start_date, n_days=n_days)
    if len(flist) == 0:
        raise ValueError("This float cycle is too old for this velocity field.")
    glorys = xr.open_mfdataset(flist, preprocess=preprocess, combine='nested', concat_dim='time', parallel=True)
    #
    return glorys


def get_armor3d_with_opendap(a_box, a_start_date, n_days=1):
    """Load ARMOR3D Multi Observation Global Ocean 3D Temperature Salinity Height Geostrophic Current and MLD

    Fields: weekly, from 1993-01-06T12:00 to current week
    Src: https://resources.marine.copernicus.eu/product-detail/MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012/INFORMATION

    If ``a_start_date+n_days`` < 2020-12-30, delivers the multi-year reprocessed (REP) weekly data, otherwise delivers near-real-time (NRT) weekly data.

    Parameters
    ----------
    a_box
    a_start_date
    n_days

    Returns
    -------
    :class:xarray.dataset
    """
    # Convert longitude to 0/360, because that's the ARMOR3D convention:
    a_box[0] = fixLON(a_box[0])
    a_box[1] = fixLON(a_box[1])

    MOTU_USERNAME, MOTU_PASSWORD = (
        os.getenv("MOTU_USERNAME"),
        os.getenv("MOTU_PASSWORD"),
    )
    if not MOTU_USERNAME:
        raise ValueError("No MOTU_USERNAME in environment ! ")

    if a_start_date + pd.Timedelta(n_days, 'D') <= pd.to_datetime('20201230'):
        serverset = 'https://nrt.cmems-du.eu/thredds/dodsC/dataset-armor-3d-rep-weekly'  # 1993-01-06 to 2020-12-30
    else:
        serverset = 'https://nrt.cmems-du.eu/thredds/dodsC/dataset-armor-3d-nrt-weekly'  # 2019-01-02 to present
    # puts("\t%s" % serverset, color=COLORS.green)

    session = requests.Session()
    session.auth = (MOTU_USERNAME, MOTU_PASSWORD)
    store = xr.backends.PydapDataStore.open(serverset, session=session)
    ds = xr.open_dataset(store)

    if a_start_date > ds['time'][-1]:
        raise ValueError("This float cycle is too young for this velocity field.\nFloat starting date %s is after the ARMOR3D last available date %s.\nTry with the GLORYS forecast." % (a_start_date, ds['time'][-1].values))

    nt = int(np.ceil(n_days / 7))
    itim = np.argwhere(ds['time'].values<a_start_date)[-1][0], \
           np.argwhere(ds['time'].values<a_start_date+(nt+1)*pd.Timedelta(7, 'D'))[-1][0]+1
    if itim[1] > len(ds['time']):
        print("Requested time frame out of max range (%s). Fall back on the longest time frame available." %
              pd.to_datetime(ds['time'][-1].values).strftime("%Y-%m-%dT%H:%M:%S"))
        itim = np.argwhere(ds['time'].values < a_start_date)[-1][0], len(ds['time'])
    idpt = np.argwhere(ds['depth'].values > 2000)[0][0]
    ilon = np.argwhere(ds['longitude'].values >= a_box[0])[0][0], np.argwhere(ds['longitude'].values >= a_box[1])[0][0]
    ilat = np.argwhere(ds['latitude'].values >= a_box[2])[0][0], np.argwhere(ds['latitude'].values >= a_box[3])[0][0]
    armor3d = ds.isel({'time': range(itim[0], itim[1]),
                       'depth': range(0, idpt),
                       'longitude': range(ilon[0], ilon[1]),
                       'latitude': range(ilat[0], ilat[1])})

    # Move back to the Argo -180/180 longitude convention:
    lon = armor3d['longitude'].values
    lon[np.argwhere(lon>180)] = lon[np.argwhere(lon>180)] - 360
    armor3d['longitude'] = lon
    return armor3d


def get_velocity_field(a_box, a_date, n_days=1, output='.', dataset='ARMOR3D'):
    """Return the velocity field as an :class:xr.Dataset, download if needed

    Parameters
    ----------
    a_box
    a_date
    n_days
    output
    dataset
    """
    velocity_file = os.path.join(output, 'velocity_%s_%idays.nc' % (dataset, n_days))
    if not os.path.exists(velocity_file):
        # Load
        if dataset == 'ARMOR3D':
            ds = get_armor3d_with_opendap(a_box, a_date, n_days=n_days)
        elif dataset == 'GLORYS':
            ds = get_glorys_with_opendap(a_box, a_date, n_days=n_days)

        # Save on file for later re-used:
        ds.to_netcdf(velocity_file)
    else:
        ds = xr.open_dataset(velocity_file)
    # print(ds)
    return ds, velocity_file


def get_HBOX(df_sim, dd=1):
    """

    Parameters
    ----------
    dd: how much to extend maps outward the deployment 'box'

    Returns
    -------
    list
    """
    rx = df_sim['deploy_lon'].max() - df_sim['deploy_lon'].min()
    ry = df_sim['deploy_lat'].max() - df_sim['deploy_lat'].min()
    lonc, latc = df_sim['deploy_lon'].mean(), df_sim['deploy_lat'].mean()
    box = [lonc - rx / 2, lonc + rx / 2, latc - ry / 2, latc + ry / 2]
    ebox = [box[i] + [-dd, dd, -dd, dd][i] for i in range(0, 4)]  # Extended 'box'

    return ebox


def get_EBOX(df_sim, df_plan, this_profile, s=1):
    """Get a box for maps

    Use all data positions from DF_SIM to make sure all points are visible
    Extend the domain by a 's' scaling factor of the deployment plan domain

    Parameters
    ----------
    s: float, default:1

    Returns
    -------
    list
    """
    box = [np.min([df_sim['deploy_lon'].min(), df_sim['longitude'].min(), df_sim['rel_lon'].min(), this_profile['longitude'].min()]),
       np.max([df_sim['deploy_lon'].max(), df_sim['longitude'].max(), df_sim['rel_lon'].max(), this_profile['longitude'].max()]),
       np.min([df_sim['deploy_lat'].min(), df_sim['latitude'].min(), df_sim['rel_lat'].min(), this_profile['latitude'].min()]),
       np.max([df_sim['deploy_lat'].max(), df_sim['latitude'].max(), df_sim['rel_lat'].max(), this_profile['latitude'].max()])]
    rx, ry = df_plan['longitude'].max() - df_plan['longitude'].min(), df_plan['latitude'].max() - df_plan['latitude'].min()
    r = np.min([rx, ry])
    ebox = [box[0]-s*r, box[1]+s*r, box[2]-s*r, box[3]+s*r]

    return ebox


def get_cfg_str(a_cfg):
    txt = "VFloat configuration: (Parking depth: %i [db], Cycle duration: %i [hours])" % (
        a_cfg.mission['parking_depth'],
        a_cfg.mission['cycle_duration']
    )
    return txt


def save_figurefile(this_fig, a_name, folder='.'):
    """

    Parameters
    ----------
    this_fig
    a_name

    Returns
    -------
    path
    """
    figname = os.path.join(folder, "%s.png" % a_name)
    log.debug("Saving %s ..." % figname)
    this_fig.savefig(figname)
    return figname


def map_add_profiles(this_ax, this_profile):
    """

    Parameters
    ----------
    this_ax

    Returns
    -------
    this_ax
    """
    this_ax.plot(this_profile['longitude'][0], this_profile['latitude'][0], 'k.', markersize=10, markeredgecolor='w')
    if this_profile.shape[0] > 1:
        this_ax.plot(this_profile['longitude'][1], this_profile['latitude'][1], 'r.', markersize=10, markeredgecolor='w')
        this_ax.arrow(this_profile['longitude'][0],
                 this_profile['latitude'][0],
                 this_profile['longitude'][1] - this_profile['longitude'][0],
                 this_profile['latitude'][1] - this_profile['latitude'][0],
                 length_includes_head=True, fc='k', ec='k', head_width=0.025, zorder=10)

    return this_ax


def map_add_features(this_ax):
    """

    Parameters
    ----------
    this_ax

    Returns
    -------
    this_ax
    """
    argoplot.utils.latlongrid(this_ax)
    this_ax.add_feature(argoplot.utils.land_feature, edgecolor="black")
    return this_ax


def map_add_cyc_nb(this_ax, this_df, lon='lon', lat='lat', cyc='cyc', pos='bt', fs=6, color='black'):
    """ Add cycle number labels next to axis

    Parameters
    ----------
    ax
    df

    Returns
    -------
    list of text label
    """
    t = []
    if pos == 'bt':
        ha, va, label = 'center', 'top', "\n{}".format
    if pos == 'tp':
        ha, va, label = 'center', 'bottom', "{}\n".format
    for irow, row in this_df.iterrows():
        this_t = this_ax.text(row[lon], row[lat], label(int(row[cyc])), ha=ha, va=va, fontsize=fs, color=color)
        t.append(this_t)
    return t

def figure_velocity(box,
                       vel, vel_name, this_profile, wmo, cyc,
                       save_figure=False, workdir='.'):
    """

    Parameters
    ----------
    box

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(box)
    ax = map_add_features(ax)
    ax = map_add_profiles(ax, this_profile)

    vel.field.isel(time=0, depth=0).plot.quiver(x="longitude", y="latitude",
                                                u=vel.var['U'], v=vel.var['V'], ax=ax, color='grey', alpha=0.5,
                                          add_guide=False)

    txt = "starting from cycle %i, predicting cycle %i" % (cyc[0], cyc[1])
    ax.set_title(
        "VirtualFleet recovery system for WMO %i: %s\n"
        "%s velocity snapshot to illustrate the simulation domain\n"
        "Vectors: Velocity field at z=%0.2fm, t=%s" %
        (wmo, txt, vel_name, vel.field['depth'][0].values[np.newaxis][0],
        pd.to_datetime(vel.field['time'][0].values).strftime("%Y/%m/%d %H:%M")), fontsize=15)

    plt.tight_layout()
    if save_figure:
        save_figurefile(fig, 'vfrecov_velocity_%s' % vel_name, workdir)
    return fig, ax


def figure_positions(this_args, vel, df_sim, df_plan, this_profile, cfg, wmo, cyc, vel_name,
                     dd=1, save_figure=False, workdir='.'):
    log.debug("Starts figure_positions")
    ebox = get_HBOX(df_sim, dd=dd)
    nfloats = df_plan.shape[0]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 7), dpi=120,
                           subplot_kw={'projection': ccrs.PlateCarree()},
                           sharex=True, sharey=True)
    ax = ax.flatten()

    for ix in [0, 1, 2]:
        ax[ix].set_extent(ebox)
        ax[ix] = map_add_features(ax[ix])

        v = vel.field.isel(time=0).interp(depth=cfg.mission['parking_depth']).plot.quiver(x="longitude",
                                                                                   y="latitude",
                                                                                   u=vel.var['U'],
                                                                                   v=vel.var['V'],
                                                                                   ax=ax[ix],
                                                                                   color='grey',
                                                                                   alpha=0.5,
                                                                                   add_guide=False)

        ax[ix].plot(df_sim['deploy_lon'], df_sim['deploy_lat'], '.', markersize=3, color='grey', alpha=0.1, markeredgecolor=None, zorder=0)
        if ix == 0:
            title = 'Velocity field at %0.2fm and deployment plan' % cfg.mission['parking_depth']
            v.set_alpha(1)
            # v.set_color('black')
        elif ix == 1:
            x, y = df_sim['longitude'], df_sim['latitude']
            title = 'Final float positions'
            sc = ax[ix].plot(x, y, '.', markersize=3, color='cyan', alpha=0.9, markeredgecolor=None)
        elif ix == 2:
            x, y = df_sim['rel_lon'], df_sim['rel_lat']
            title = 'Final floats position relative to last float position'
            sc = ax[ix].plot(x, y, '.', markersize=3, color='cyan', alpha=0.9, markeredgecolor=None)

        ax[ix] = map_add_profiles(ax[ix], this_profile)
        ax[ix].set_title(title)

    fig.suptitle("VirtualFleet recovery prediction for WMO %i: starting from cycle %i, predicting cycle %i\n%s" %
                 (wmo, cyc[0], cyc[1], get_cfg_str(cfg)), fontsize=15)
    plt.tight_layout()
    if save_figure:
        save_figurefile(fig, "vfrecov_positions_%s" % get_sim_suffix(this_args, cfg), workdir)
    return fig, ax


def figure_predictions(this_args, weights, bin_X, bin_Y, bin_res, Hrel, recovery,
                       vel, df_sim, df_plan, this_profile, cfg, wmo, cyc, vel_name,
                       s=0.2, alpha=False, save_figure=False, workdir='.'):
    log.debug("Starts figure_predictions")
    ebox = get_EBOX(df_sim, df_plan, this_profile, s=s)
    nfloats = df_plan.shape[0]
    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,7), dpi=90,
    #                        subplot_kw={'projection': ccrs.PlateCarree()},
    #                        sharex=True, sharey=True)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 15), dpi=120,
                           subplot_kw={'projection': ccrs.PlateCarree()},
                           sharex=True, sharey=True)
    ax = ax.flatten()

    xpred, ypred = recovery['prediction_location']['longitude']['value'], \
                   recovery['prediction_location']['latitude']['value']

    for ix in [0, 1, 2, 3]:
        # log.debug("Plot %i" % ix)
        ax[ix].set_extent(ebox)
        ax[ix] = map_add_features(ax[ix])

        vel.field.isel(time=0).interp(depth=cfg.mission['parking_depth']).plot.quiver(x="longitude",
                                                                                   y="latitude",
                                                                                   u=vel.var['U'],
                                                                                   v=vel.var['V'],
                                                                                   ax=ax[ix],
                                                                                   color='grey',
                                                                                   alpha=0.5,
                                                                                   scale=1,
                                                                                   add_guide=False)

        ax[ix].plot(df_sim['deploy_lon'], df_sim['deploy_lat'], '.',
                    markersize=3,
                    color='grey',
                    alpha=0.1,
                    markeredgecolor=None,
                    zorder=0)
        w = weights/np.max(np.abs(weights), axis=0)
        ii = np.argsort(w)
        cmap = plt.cm.cool
        # cmap = plt.cm.Reds
        if ix == 0:
            x, y = df_sim['deploy_lon'], df_sim['deploy_lat']
            # title = 'Initial float positions\ncolored with histogram weights'
            title = 'Initial virtual float positions'
            # wp = weights_plan/np.nanmax(np.abs(weights_plan),axis=0)
            if not alpha:
                sc = ax[ix].scatter(x[ii], y[ii], c=w[ii], marker='o', s=4, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
            else:
                sc = ax[ix].scatter(x[ii], y[ii], c=w[ii], marker='o', s=4, alpha=w[ii], edgecolor=None, vmin=0, vmax=1, cmap=cmap)
        elif ix == 1:
            x, y = df_sim['longitude'], df_sim['latitude']
            # title = 'Final float positions\ncolored with histogram weights'
            title = 'Final virtual float positions'
            if not alpha:
                sc = ax[ix].scatter(x, y, c=w, marker='o', s=4, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
            else:
                sc = ax[ix].scatter(x, y, c=w, marker='o', s=4, alpha=w, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
        elif ix == 2:
            x, y = df_sim['rel_lon'], df_sim['rel_lat']
            # title = 'Final floats relative to last float position\ncolored with histogram weights'
            title = 'Final virtual floats positions relative to observed float'
            if not alpha:
                sc = ax[ix].scatter(x[ii], y[ii], c=w[ii], marker='o', s=4, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
            else:
                sc = ax[ix].scatter(x[ii], y[ii], c=w[ii], marker='o', s=4, alpha=w[ii], edgecolor=None, vmin=0, vmax=1, cmap=cmap)
        elif ix == 3:
            # Hs = H/(np.nanmax(H)-np.nanmin(H))
            # Hs = Hrel/(np.nanmax(Hrel)-np.nanmin(Hrel))
            sc = ax[ix].pcolor(bin_X[0:-1]+bin_res/2, bin_Y[0:-1]+bin_res/2, Hrel.T, cmap=cmap)
            # bin_X, bin_Y = np.meshgrid(bin_X[0:-1]+bin_res/2, bin_Y[0:-1]+bin_res/2)
            # bin_X, bin_Y = bin_X.flatten(), bin_Y.flatten()
            # c = (Hrel.T).flatten()
            # alp = c/np.nanmax(np.abs(c),axis=0)
            # alp[np.isnan(alp)] = 0
            # sc = ax[ix].scatter(bin_X, bin_Y, c=c, marker='o', s=6, alpha=alp, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
            title = 'Weighted profile density'

        # Trajectory prediction:
        ax[ix].arrow(this_profile['longitude'][0],
                     this_profile['latitude'][0],
                     xpred-this_profile['longitude'][0],
                     ypred-this_profile['latitude'][0],
                     length_includes_head=True, fc='k', ec='c', head_width=0.025, zorder=10)
        ax[ix].plot(xpred, ypred, 'k+', zorder=10)

        # distance_due_to_timelag
        if this_profile.shape[0] > 1:
            km2deg = 360 / (2 * np.pi * 6371) # Approximation
            ax[ix].add_patch(
                mpatches.Circle(xy=[xpred, ypred],
                                radius=recovery['prediction_metrics']['surface_drift']['value'] * km2deg,
                                color='green',
                                alpha=0.7,
                                transform=ccrs.PlateCarree(),
                                zorder=9))

        # Another
        # xave, yave = np.average(DF_SIM['longitude'].values, weights=weights), \
        #              np.average(DF_SIM['latitude'].values, weights=weights)
        # # ax[ix].arrow(THIS_PROFILE['longitude'][0],
        # #              THIS_PROFILE['latitude'][0],
        # #              xave-THIS_PROFILE['longitude'][0],
        # #              yave-THIS_PROFILE['latitude'][0],
        # #              length_includes_head=True, fc='k', ec='c', head_width=0.025, zorder=10)
        # ax[ix].plot([THIS_PROFILE['longitude'][0], xave], [THIS_PROFILE['latitude'][0], yave], 'g--', zorder=10)

        plt.colorbar(sc, ax=ax[ix],shrink =0.5)

        ax[ix] = map_add_profiles(ax[ix], this_profile)
        ax[ix].set_title(title)

    log.debug("Start to write metrics string")

    err = recovery['prediction_location_error']
    met = recovery['prediction_metrics']
    if this_profile.shape[0] > 1:
        # err_str = "Prediction vs Truth: [%0.2fkm, $%0.2f^o$]" % (err['distance'], err['bearing'])
        err_str = "Prediction errors: [dist=%0.2f%s, bearing=$%0.2f^o$, time=%s]\n" \
                  "Distance error represents %s of transit at 12kt" % (err['distance']['value'],
                                                  err['distance']['unit'],
                                                  err['bearing']['value'],
                                                  strfdelta(pd.Timedelta(err['time']['value'], 'h'),
                                                            "{hours}H{minutes:02d}"),
                                                  strfdelta(pd.Timedelta(met['transit']['value'], 'h'),
                                                            "{hours}H{minutes:02d}"))
    else:
        err_str = ""

    fig.suptitle("VirtualFleet recovery prediction for WMO %i: \
    starting from cycle %i, predicting cycle %i\n%s\n%s\n%s" %
                 (wmo, cyc[0], cyc[1], get_cfg_str(cfg), err_str, "Prediction based on %s" % vel_name), fontsize=15)
    plt.tight_layout()
    if save_figure:
        save_figurefile(fig, 'vfrecov_predictions_%s' % get_sim_suffix(this_args, cfg), workdir)
    return fig, ax


def figure_predictions_recap(this_args, weights, bin_X, bin_Y, bin_res, Hrel, recovery,
                       vel, df_sim, df_plan, this_profile, cfg, wmo, cyc, vel_name,
                       s=0.2, alpha=False, save_figure=False, workdir='.'):
    log.debug("Starts figure_predictions_recap")
    ebox = get_EBOX(df_sim, df_plan, this_profile, s=s)
    nfloats = df_plan.shape[0]
    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,7), dpi=90,
    #                        subplot_kw={'projection': ccrs.PlateCarree()},
    #                        sharex=True, sharey=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 10), dpi=90,
                           subplot_kw={'projection': ccrs.PlateCarree()},
                           sharex=True, sharey=True)
    ax, ix = np.array(ax)[np.newaxis], 0

    xpred, ypred = recovery['prediction_location']['longitude']['value'], \
                   recovery['prediction_location']['latitude']['value']

    ax[ix].set_extent(ebox)
    ax[ix] = map_add_features(ax[ix])

    vel.field.isel(time=0).interp(depth=cfg.mission['parking_depth']).plot.quiver(x="longitude",
                                                                                  y="latitude",
                                                                                  u=vel.var['U'],
                                                                                  v=vel.var['V'],
                                                                                  ax=ax[ix],
                                                                                  color='grey',
                                                                                  alpha=0.5,
                                                                                  scale=1,
                                                                                  add_guide=False)

    ax[ix].plot(df_sim['deploy_lon'], df_sim['deploy_lat'], '.',
                markersize=3,
                color='grey',
                alpha=0.1,
                markeredgecolor=None,
                zorder=0)

    w = weights/np.max(np.abs(weights), axis=0)
    ii = np.argsort(w)
    cmap = plt.cm.cool

    x, y = df_sim['rel_lon'], df_sim['rel_lat']
    title = 'Final virtual floats positions relative to observed starting float'
    if not alpha:
        sc = ax[ix].scatter(x[ii], y[ii], c=w[ii], marker='o', s=4, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
    else:
        sc = ax[ix].scatter(x[ii], y[ii], c=w[ii], marker='o', s=4, alpha=w[ii], edgecolor=None, vmin=0, vmax=1, cmap=cmap)

    # Trajectory prediction:
    ax[ix].arrow(this_profile['longitude'][0],
                 this_profile['latitude'][0],
                 xpred-this_profile['longitude'][0],
                 ypred-this_profile['latitude'][0],
                 length_includes_head=True, fc='k', ec='r', head_width=0.025, zorder=10)
    ax[ix].plot(xpred, ypred, 'k+', zorder=10)

    # distance_due_to_timelag
    if this_profile.shape[0] > 1:
        km2deg = 360 / (2 * np.pi * 6371) # Approximation
        ax[ix].add_patch(
            mpatches.Circle(xy=[xpred, ypred],
                            radius=recovery['prediction_metrics']['surface_drift']['value'] * km2deg,
                            color='green',
                            alpha=0.4,
                            transform=ccrs.PlateCarree(),
                            zorder=9))

    plt.colorbar(sc, ax=ax[ix], shrink=0.5, label='Norm. gaussian distance to starting position')

    ax[ix] = map_add_profiles(ax[ix], this_profile)
    ax[ix].set_title('')

    err = recovery['prediction_location_error']
    met = recovery['prediction_metrics']
    if this_profile.shape[0] > 1:
        # err_str = "Prediction vs Truth: [%0.2fkm, $%0.2f^o$]" % (err['distance'], err['bearing'])
        err_str = "Prediction errors: [dist=%0.2f%s, bearing=$%0.2f^o$, time=%s], " \
                  "Distance error represents %s of transit at 12kt" % (err['distance']['value'],
                                                  err['distance']['unit'],
                                                  err['bearing']['value'],
                                                  strfdelta(pd.Timedelta(err['time']['value'], 'h'),
                                                            "{hours}H{minutes:02d}"),
                                                  strfdelta(pd.Timedelta(met['transit']['value'], 'h'),
                                                            "{hours}H{minutes:02d}"))
    else:
        err_str = ""

    fig.suptitle("VirtualFleet recovery prediction for WMO %i: \
starting from cycle %i, predicting cycle %i\n%s\n%s\n%s\nFigure: %s" %
                 (wmo, cyc[0], cyc[1], get_cfg_str(cfg), err_str, "Prediction based on %s" % vel_name, title), fontsize=12)
    plt.tight_layout()
    if save_figure:
        save_figurefile(fig, 'vfrecov_predictions_recap_%s' % get_sim_suffix(this_args, cfg), workdir)
    return fig, ax


def setup_deployment_plan(a_profile, a_date, nfloats=15000):
    # We will deploy a collection of virtual floats that are located around the real float with random perturbations in space and time

    # Amplitude of the profile position perturbations in the zonal (deg), meridional (deg), and temporal (hours) directions:
    rx = 0.5
    ry = 0.5
    rt = 0

    #
    lonc, latc = a_profile
    # box = [lonc - rx / 2, lonc + rx / 2, latc - ry / 2, latc + ry / 2]

    a, b = lonc - rx / 2, lonc + rx / 2
    lon = (b - a) * np.random.random_sample((nfloats,)) + a

    a, b = latc - ry / 2, latc + ry / 2
    lat = (b - a) * np.random.random_sample((nfloats,)) + a

    a, b = 0, rt
    dtim = (b - a) * np.random.random_sample((nfloats,)) + a
    dtim = np.round(dtim).astype(int)
    tim = pd.to_datetime([a_date + np.timedelta64(dt, 'h') for dt in dtim])
    # dtim = (b-a) * np.random.random_sample((nfloats, )) + a
    # dtim = np.round(dtim).astype(int)
    # tim2 = pd.to_datetime([this_date - np.timedelta64(dt, 'h') for dt in dtim])
    # tim = np.sort(np.concatenate([tim2, tim1]))

    # Round time to the o(5mins), same as step=timedelta(minutes=5) in the simulation params
    tim = tim.round(freq='5T')

    #
    df = pd.DataFrame(
        [tim, lat, lon, np.arange(0, nfloats) + 9000000, np.full_like(lon, 0), ['VF' for l in lon], ['?' for l in lon]],
        index=['date', 'latitude', 'longitude', 'wmo', 'cycle_number', 'institution_code', 'file']).T

    return df


def simu2index_legacy(df_plan, this_ds):
    # Specific method for the recovery simulations
    # This is very slow but could be optimized
    ds_list = []
    for irow_plan, row_plan in tqdm(df_plan.iterrows()):
        ii = np.argwhere(((this_ds['lon'].isel(obs=0) - row_plan['longitude']) ** 2 + (
                    this_ds['lat'].isel(obs=0) - row_plan['latitude']) ** 2).values == 0)
        if ii:
            wmo = row_plan['wmo']
            deploy_lon, deploy_lat = row_plan['longitude'], row_plan['latitude']
            itraj = ii[0][0]
            for cyc, grp in this_ds.isel(traj=itraj).groupby(group='cycle_number'):
                ds_cyc = grp.isel(obs=-1)
                if cyc == 1:
                    if ds_cyc['cycle_phase'] in [3, 4]:
                        ds_cyc['wmo'] = xr.DataArray(np.full_like((1,), fill_value=wmo), dims='obs')
                        ds_cyc['deploy_lon'] = xr.DataArray(np.full_like((1,), fill_value=deploy_lon, dtype=float),
                                                            dims='obs')
                        ds_cyc['deploy_lat'] = xr.DataArray(np.full_like((1,), fill_value=deploy_lat, dtype=float),
                                                            dims='obs')
                        ds_list.append(ds_cyc)

    ds_profiles = xr.concat(ds_list, dim='obs')
    df = ds_profiles.to_dataframe()
    df = df.rename({'time': 'date', 'lat': 'latitude', 'lon': 'longitude', 'z': 'min_depth'}, axis='columns')
    df = df[['date', 'latitude', 'longitude', 'wmo', 'cycle_number', 'deploy_lon', 'deploy_lat']]
    df['wmo'] = df['wmo'].astype('int')
    df = df.reset_index(drop=True)
    return df


def ds_simu2index(this_ds):
    # Instead of really looking at the cycle phase and structure, we just pick the last trajectory point from output file !
    # This is way much faster and a good approximation IF the simulation length is the cycling frequency.
    data = {
        'date': this_ds.isel(obs=-1)['time'],
        'latitude': this_ds.isel(obs=-1)['lat'],
        'longitude': this_ds.isel(obs=-1)['lon'],
        'wmo': 9000000 + this_ds['traj'].values,
        'deploy_lon': this_ds.isel(obs=0)['lon'],
        'deploy_lat': this_ds.isel(obs=0)['lat']
    }
    df = pd.DataFrame(data)
    df['wmo'] = df['wmo'].astype(int)
    return df


def get_index(vf, vel, df_plan):
    # Instead of really looking at the cycle phase and structure, we just pick the last trajectory point in memory !
    # This is way much faster and a good approximation IF the simulation length is the cycling frequency.
    data = {
        'date': [vel.field['time'][0].values + pd.Timedelta(dt, 's') for dt in vf.ParticleSet.time],
        'latitude': vf.ParticleSet.lat,
        'longitude': vf.ParticleSet.lon,
        'wmo': 9000000 + np.arange(0, vf.ParticleSet.lon.shape[0]),
        'deploy_lon': df_plan['longitude'],
        'deploy_lat': df_plan['latitude']
    }
    if len(data['latitude']) != len(data['deploy_lat']):
        raise ValueError('Virtual floats have been lost during the simulation ! %i simulated vs %i deployed' % (len(data['latitude']), len(data['deploy_lat'])))
    df = pd.DataFrame(data)
    df['wmo'] = df['wmo'].astype(int)
    return df


def postprocess_index(this_df, this_profile):
    # Compute some distances

    # Compute distance between the predicted profile and the initial profile location from the deployment plan
    # We assume that virtual floats are sequentially taken from the deployment plan
    # Since distances are very short, we compute a simple rectangular distance

    x2, y2 = this_profile['longitude'].values[0], this_profile['latitude'].values[0]  # real float initial position
    this_df['distance'] = np.nan
    this_df['rel_lon'] = np.nan
    this_df['rel_lat'] = np.nan
    this_df['distance_origin'] = np.nan

    for isim, sim_row in this_df.iterrows():
        # Profile coordinates:
        x0, y0 = sim_row['deploy_lon'], sim_row['deploy_lat']  # virtual float initial position
        x1, y1 = sim_row['longitude'], sim_row['latitude']  # virtual float final position

        # Distance between each pair of cycles of virtual floats:
        dist = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
        this_df.loc[isim, 'distance'] = dist

        # Shift between each pair of cycles:
        dx, dy = x1 - x0, y1 - y0
        # Get a relative displacement from real float initial position:
        this_df.loc[isim, 'rel_lon'] = x2 + dx
        this_df.loc[isim, 'rel_lat'] = y2 + dy

        # Distance between the predicted profile and the observed initial profile
        dist = np.sqrt((y2 - y0) ** 2 + (x2 - x0) ** 2)
        # dinit.append(dist)
        this_df.loc[isim, 'distance_origin'] = dist

    return this_df


def predict_position(this_args, workdir, wmo, cyc, cfg, vel, vel_name, df_sim, df_plan, this_profile,
                     save_figure=False, quiet=False):
    """ Compute the position of the next profile for recovery

    Prediction is based on weighted statistics from the last position of virtual floats

    Returns
    -------
    dict

    """

    def get_weights(scale=20):
        """Return weights as a gaussian distance with a std based on the size of the deployment domain"""
        rx, ry = df_plan['longitude'].max() - df_plan['longitude'].min(), \
                 df_plan['latitude'].max() - df_plan['latitude'].min()
        r = np.min([rx, ry])  # Minimal size of the deployment domain
        weights = np.exp(-(df_sim['distance_origin'] ** 2) / (r / scale))
        weights[np.isnan(weights)] = 0
        return weights

    # Compute a weighted histogram of the virtual float positions
    hbox = get_EBOX(df_sim, df_plan, this_profile, s=1)
    bin_res = 1 / 12 / 2
    bin_x, bin_y = np.arange(hbox[0], hbox[1], bin_res), np.arange(hbox[2], hbox[3], bin_res)
    weights = get_weights(scale=20)
    Hrel, xedges, yedges = np.histogram2d(df_sim['rel_lon'],
                                          df_sim['rel_lat'],
                                          bins=[bin_x, bin_y],
                                          weights=weights,
                                          density=True)

    # Get coordinates of the most probable location (max of the histogram):
    ixmax, iymax = np.unravel_index(Hrel.argmax(), Hrel.shape)
    xpred, ypred = (bin_x[0:-1] + bin_res / 2)[ixmax], (bin_y[0:-1] + bin_res / 2)[iymax]
    tpred = df_sim['date'].mean()
    recovery = {'prediction_location': {'longitude': {'value': xpred, 'unit': 'degree East'},
                                        'latitude': {'value': ypred, 'unit': 'degree North'},
                                        'time': {'value': tpred}}}

    # Nicer histogram
    Hrel[Hrel == 0] = np.NaN

    # Compute error metrics of the predicted position:
    error = {'distance': {'value': None,
                          'unit': 'km'},
             'bearing': {'value': None,
                         'unit': 'degree'},
             'time': {'value': None,
                      'unit': 'hour'}
             }
    if this_profile.shape[0] > 1:
        dd = haversine(this_profile['longitude'][1], this_profile['latitude'][1], xpred, ypred)
        dt = pd.Timedelta(recovery['prediction_location']['time']['value']-
                          this_profile['date'].values[-1]).seconds/3600.
        error['distance']['value'] = dd
        error['bearing']['value'] = bearing(this_profile['longitude'][0],
                                    this_profile['latitude'][0],
                                    this_profile['longitude'][1],
                                    this_profile['latitude'][1]) - bearing(this_profile['longitude'][0],
                                                                           this_profile['latitude'][0],
                                                                           xpred,
                                                                           ypred)
        error['time']['value'] = dt

    # Compute more metrics to understand the prediction:
    metrics = {}
    # Compute a transit time to cover the distance error:
    # (assume a 12 kts boat speed with 1 kt = 1.852 km/h)
    metrics['transit'] = {'value': None,
                          'unit': 'hour',
                          'comment': 'Boat transit time to cover the distance error '
                                     '(assume a 12 kts boat speed with 1 kt = 1.852 km/h)'}
    if error['distance']['value'] is not None:
        metrics['transit']['value'] = pd.Timedelta(error['distance']['value'] / (12 * 1.852), 'h').seconds/3600.

    # Compute the possible drift due to the time lag between the predicted profile timing and the expected one:
    dsc = vel.field.interp(
        {vel.dim['lon']: xpred,
         vel.dim['lat']: ypred,
         vel.dim['time']: tpred,
         vel.dim['depth']: vel.field[{vel.dim['depth']: 0}][vel.dim['depth']].values[np.newaxis][0]}
    )
    velc = np.sqrt(dsc[vel.var['U']] ** 2 + dsc[vel.var['V']] ** 2).values[np.newaxis][0]
    metrics['surface_drift'] = {'value': None,
                                'unit': 'km',
                                'surface_currents_speed': None,
                                'surface_currents_speed_unit': 'm/s',
                                'comment': 'Drift by surface currents due to the float ascent time error '
                                           '(difference between simulated profile time and the observed one).'}
    if error['time']['value'] is not None:
        metrics['surface_drift']['value'] = (error['time']['value']*3600 * velc / 1e3)
        metrics['surface_drift']['surface_currents_speed'] = velc

    # Add to the mean result dict:
    recovery['prediction_location_error'] = error
    recovery['prediction_metrics'] = metrics

    # Final figures:
    fig, ax = figure_predictions(this_args, weights, bin_x, bin_y, bin_res, Hrel, recovery,
                                 vel, df_sim, df_plan, this_profile, cfg, wmo, cyc, vel_name,
                                 save_figure=save_figure, workdir=workdir)

    fig, ax = figure_predictions_recap(this_args, weights, bin_x, bin_y, bin_res, Hrel, recovery,
                                 vel, df_sim, df_plan, this_profile, cfg, wmo, cyc, vel_name,
                                 save_figure=save_figure, workdir=workdir)

    #
    return recovery


def analyse_pairwise_distances(this_args, this_cfg,  data):
    from scipy.signal import find_peaks

    def get_hist_and_peaks(this_d):
        x = this_d.flatten()
        x = x[~np.isnan(x)]
        x = x[:, np.newaxis]
        hist, bin_edges = np.histogram(x, bins=100, density=1)
        # dh = np.diff(bin_edges[0:2])
        peaks, _ = find_peaks(hist / np.max(hist), height=.4, distance=20)
        return {'pdf': hist, 'bins': bin_edges[0:-1], 'Npeaks': len(peaks)}

    # Trajectory file:
    workdir = this_args.output
    simufile = os.path.sep.join([workdir,
                               'trajectories_%s.zarr' % get_sim_suffix(this_args, this_cfg)])
    engine = "zarr"
    if not os.path.exists(simufile):
        ncfile = os.path.sep.join([workdir,
                                   'trajectories_%s.nc' % get_sim_suffix(this_args, this_cfg)])
        engine = "netcdf4"
        if not os.path.exists(ncfile):
            puts('Cannot analyse pairwise distances because the trajectory file cannot be found at: %s' % simufile,
                 color=COLORS.red)
            return data  # Return results dict unchanged

    # Open trajectory file:
    ds = xr.open_dataset(simufile, engine=engine)

    # Compute trajectories relative to the single/only real float initial position:
    lon = ds['lon'].values
    lat = ds['lat'].values
    lon0 = data['previous_profile']['location']['longitude']['value']
    lat0 = data['previous_profile']['location']['latitude']['value']
    ds['lonc'] = xr.DataArray(lon - np.broadcast_to(lon[:, 0][:, np.newaxis], lon.shape) + lon0, dims=['trajectory', 'obs'])
    ds['latc'] = xr.DataArray(lat - np.broadcast_to(lat[:, 0][:, np.newaxis], lat.shape) + lat0, dims=['trajectory', 'obs'])

    # Compute trajectory lengths:
    ds['length'] = np.sqrt(ds.diff(dim='obs')['lon'] ** 2 + ds.diff(dim='obs')['lat'] ** 2).sum(dim='obs')
    ds['lengthc'] = np.sqrt(ds.diff(dim='obs')['lonc'] ** 2 + ds.diff(dim='obs')['latc'] ** 2).sum(dim='obs')

    # Compute initial points pairwise distances, PDF and nb of peaks:
    X = ds.isel(obs=0)
    X = X.isel(trajectory=~np.isnan(X['lon']))
    X0 = np.array((X['lon'].values, X['lat'].values)).T
    d0 = pairwise_distances(X0, n_jobs=-1)
    d0 = np.triu(d0)
    d0[d0 == 0] = np.nan

    x0 = d0.flatten()
    x0 = x0[~np.isnan(x0)]
    x0 = x0[:, np.newaxis]

    hist0, bin_edges0 = np.histogram(x0, bins=100, density=1)
    dh0 = np.diff(bin_edges0[0:2])
    peaks0, _ = find_peaks(hist0 / np.max(hist0), height=.4, distance=20)

    # Compute final points pairwise distances, PDF and nb of peaks:
    X = ds.isel(obs=-1)
    X = X.isel(trajectory=~np.isnan(X['lon']))
    dsf = X
    X = np.array((X['lon'].values, X['lat'].values)).T
    d = pairwise_distances(X, n_jobs=-1)
    d = np.triu(d)
    d[d == 0] = np.nan

    x = d.flatten()
    x = x[~np.isnan(x)]
    x = x[:, np.newaxis]

    hist, bin_edges = np.histogram(x, bins=100, density=1)
    dh = np.diff(bin_edges[0:2])
    peaks, _ = find_peaks(hist / np.max(hist), height=.4, distance=20)

    # Compute final points pairwise distances (relative traj), PDF and nb of peaks:
    X1 = ds.isel(obs=-1)
    X1 = X1.isel(trajectory=~np.isnan(X1['lonc']))
    dsfc = X1
    X1 = np.array((X1['lonc'].values, X1['latc'].values)).T
    d1 = pairwise_distances(X1, n_jobs=-1)
    d1 = np.triu(d1)
    d1[d1 == 0] = np.nan

    x1 = d1.flatten()
    x1 = x1[~np.isnan(x1)]
    x1 = x1[:, np.newaxis]

    hist1, bin_edges1 = np.histogram(x1, bins=100, density=1)
    dh1 = np.diff(bin_edges1[0:2])
    peaks1, _ = find_peaks(hist1 / np.max(hist1), height=.4, distance=20)

    # Compute the overlapping between the initial and relative state PDFs:
    bin_unif = np.arange(0, np.max([bin_edges0, bin_edges1]), np.min([dh0, dh1]))
    dh_unif = np.diff(bin_unif[0:2])
    hist0_unif = np.interp(bin_unif, bin_edges0[0:-1], hist0)
    hist_unif = np.interp(bin_unif, bin_edges[0:-1], hist)
    hist1_unif = np.interp(bin_unif, bin_edges1[0:-1], hist1)

    # Area under hist1 AND hist0:
    # overlapping = np.sum(hist1_unif[hist0_unif >= hist1_unif]*dh_unif)
    overlapping = np.sum(hist_unif[hist0_unif >= hist_unif] * dh_unif)

    # Ratio of the max PDF ranges:
    # staggering = np.max(bin_edges1)/np.max(bin_edges0)
    staggering = np.max(bin_edges) / np.max(bin_edges0)

    # Save metrics:
    prediction_metrics = data['prediction_metrics']
    # prediction_metrics = {}

    prediction_metrics['trajectory_lengths'] = {'median': np.nanmedian(ds['length'].values),
                                                'std': np.nanstd(ds['length'].values)}

    prediction_metrics['pairwise_distances'] = {
        'initial_state': {'median': np.nanmedian(d0), 'std': np.nanstd(d0), 'nPDFpeaks': len(peaks0)},
        'final_state': {'median': np.nanmedian(d), 'std': np.nanstd(d), 'nPDFpeaks': len(peaks)},
        'relative_state': {'median': np.nanmedian(d1), 'std': np.nanstd(d1), 'nPDFpeaks': len(peaks1)},
        'overlapping': {'value': overlapping,
                        'comment': 'Overlapping area between PDF(initial_state) and PDF(final_state)'},
        'staggering': {'value': staggering, 'comment': 'Ratio of PDF(initial_state) vs PDF(final_state) ranges'},
        'score': {'value': overlapping / len(peaks), 'comment': 'overlapping/nPDFpeaks(final_state)'}}

    if np.isinf(overlapping / len(peaks)):
        raise ValueError("Can't compute the prediction score, infinity !")

    ratio = prediction_metrics['pairwise_distances']['final_state']['std'] / \
            prediction_metrics['pairwise_distances']['initial_state']['std']
    prediction_metrics['pairwise_distances']['std_ratio'] = ratio

    data['prediction_metrics'] = prediction_metrics

    # Figure:
    backend = matplotlib.get_backend()
    if this_args.json:
        matplotlib.use('Agg')

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 10), dpi=90)
    ax, ix = ax.flatten(), -1
    cmap = plt.cm.coolwarm

    ix += 1
    dd = dsf['length'].values
    ax[ix].plot(X0[:, 0], X0[:, 1], '.', markersize=3, color='grey', alpha=0.5, markeredgecolor=None, zorder=0)
    ax[ix].scatter(X[:, 0], X[:, 1], c=dd, zorder=10, s=3, cmap=cmap)
    ax[ix].grid()
    this_traj = int(dsf.isel(trajectory=np.argmax(dd))['trajectory'].values[np.newaxis][0])
    ax[ix].plot(ds.where(ds['trajectory'] == this_traj, drop=True).isel(trajectory=0)['lon'],
                ds.where(ds['trajectory'] == this_traj, drop=True).isel(trajectory=0)['lat'], 'r',
                zorder=13, label='Longest traj.')
    this_traj = int(dsf.isel(trajectory=np.argmin(dd))['trajectory'].values[np.newaxis][0])
    ax[ix].plot(ds.where(ds['trajectory'] == this_traj, drop=True).isel(trajectory=0)['lon'],
                ds.where(ds['trajectory'] == this_traj, drop=True).isel(trajectory=0)['lat'], 'b',
                zorder=13, label='Shortest traj.')
    ax[ix].legend()
    ax[ix].set_title('Trajectory lengths')

    ix += 1
    ax[ix].plot(bin_edges0[0:-1], hist0, label='Initial (%i peak)' % len(peaks0), color='gray')
    ax[ix].plot(bin_edges[0:-1], hist, label='Final (%i peak)' % len(peaks), color='lightblue')
    ax[ix].plot(bin_edges[peaks], hist[peaks], "x", label='Peaks')
    ax[ix].legend()
    ax[ix].grid()
    ax[ix].set_xlabel('Pairwise distance [degree]')
    line1 = "Staggering: %0.4f" % staggering
    line2 = "Overlapping: %0.4f" % overlapping
    line3 = "Score: %0.4f" % (overlapping / len(peaks))
    ax[ix].set_title("Pairwise distances PDF: [%s / %s / %s]" % (line1, line2, line3))

    line0 = "VirtualFleet recovery prediction for WMO %i: starting from cycle %i, predicting cycle %i\n%s" % \
            (this_args.wmo, this_args.cyc - 1, this_args.cyc, get_cfg_str(this_cfg))
    line1 = "Prediction made with %s and %i virtual floats" % (this_args.velocity, this_args.nfloats)
    fig.suptitle("%s\n%s" % (line0, line1), fontsize=15)
    plt.tight_layout()
    if this_args.save_figure:
        figfile = 'vfrecov_metrics01_%s' % get_sim_suffix(this_args, this_cfg)
        save_figurefile(fig, figfile, workdir)

    # Save new data to json file:
    # jsfile = os.path.join(workdir, 'prediction_%s_%i.json' % (this_args.velocity, this_args.nfloats))
    # with open(jsfile, 'w', encoding='utf-8') as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4, default=str, sort_keys=True)

    if this_args.json:
        matplotlib.use(backend)
    return data


def setup_args():
    icons_help_string = """This script can be used to make prediction of a specific float cycle position.
    This script can be used on past or unknown float cycles.
    Note that in order to download online velocity field from 'https://nrt.cmems-du.eu', you need to set the environment variables: MOTU_USERNAME and MOTU_PASSWORD.
        """

    parser = argparse.ArgumentParser(description='VirtualFleet recovery predictor',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="%s\n(c) Argo-France/Ifremer/LOPS, 2022" % icons_help_string)

    # Add long and short arguments
    parser.add_argument('wmo', help="Float WMO number", type=int)
    parser.add_argument("cyc", help="Cycle number to predict", type=int)
    parser.add_argument("--nfloats", help="Number of virtual floats used to make the prediction, default: 2000",
                        type=int, default=2000)
    parser.add_argument("--output", help="Output folder, default: webAPI internal folder", default=None)
    parser.add_argument("--velocity", help="Velocity field to use. Possible values are: 'ARMOR3D' (default), 'GLORYS'",
                        default='ARMOR3D')
    parser.add_argument("--save_figure", help="Should we save figure on file or not ? Default: True", default=True)
    parser.add_argument("--save_sim", help="Should we save the simulation on file or not ? Default: False", default=False)
    parser.add_argument("--vf", help="Parent folder to the VirtualFleet repository clone", default=None)
    parser.add_argument("--json", help="Use to only return a json file and stay quiet", action='store_true')

    parser.add_argument("--cfg_parking_depth", help="Virtual floats parking depth in [db], default: use previous cycle value", default=None)
    parser.add_argument("--cfg_cycle_duration", help="Virtual floats cycle duration in [hours], default: use previous cycle value", default=None)

    return parser


def get_sim_suffix(this_args, this_cfg):
    """Compose a string suffix for output files"""
    # suf = '%s_%i' % (this_args.velocity, this_args.nfloats)
    suf = 'VEL%s_NF%i_CYCDUR%i_PDPTH%i' % (this_args.velocity,
                                                this_args.nfloats,
                                                int(this_cfg.mission['cycle_duration']),
                                                int(this_cfg.mission['parking_depth']))
    return suf


def predictor(args):
    """Prediction manager"""
    execution_start = time.time()
    process_start = time.process_time()

    if is_wmo(args.wmo):
        WMO = args.wmo
    if is_cyc(args.cyc) and args.cyc > 1:
        CYC = [args.cyc-1, args.cyc]
    if args.velocity not in ['ARMOR3D', 'GLORYS']:
        raise ValueError("Velocity field must be one in: ['ARMOR3D', 'GLORYS']")
    else:
        VEL_NAME = args.velocity.upper()

    if args.save_figure:
        mplbackend = matplotlib.get_backend()
        matplotlib.use('Agg')

    # Where do we find the VirtualFleet repository ?
    if not args.vf:
        euroargodev = os.path.expanduser('~/git/github/euroargodev')
    else:
        euroargodev = os.path.abspath(args.vf)
        if not os.path.exists(os.path.join(euroargodev, "VirtualFleet")):
            raise ValueError("VirtualFleet can't be found at '%s'" % euroargodev)

    # Import the VirtualFleet library
    sys.path.insert(0, os.path.join(euroargodev, "VirtualFleet"))
    from virtualargofleet import VelocityField, VirtualFleet, FloatConfiguration
    # from virtualargofleet.app_parcels import ArgoParticle

    # Set-up the working directory:
    if not args.output:
        WORKDIR = os.path.sep.join([get_package_dir(), "webapi", "myapp", "static", "data", str(WMO), str(CYC[1])])
    else:
        WORKDIR = os.path.sep.join([args.output, str(WMO), str(CYC[1])])
    WORKDIR = os.path.abspath(WORKDIR)
    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)
    args.output = WORKDIR

    if not args.json:
        puts("\nData will be saved in:")
        puts("\t%s" % WORKDIR, color=COLORS.green)

    # Set-up logger
    logging.basicConfig(
        level=logging.DEBUG,
        format=DEBUGFORMATTER,
        datefmt='%m/%d/%Y %I:%M:%S %p',
        handlers=[logging.FileHandler(os.path.join(WORKDIR, "vfpred.log"), mode='a')]
    )

    # Load these profiles information:
    if not args.json:
        puts("\nYou can check this float dashboard while we prepare the prediction:")
        puts("\t%s" % argoplot.dashboard(WMO, url_only=True), color=COLORS.green)
    host = "/home/ref-argo/gdac" if not os.uname()[0] == 'Darwin' else "https://data-argo.ifremer.fr"
    # host = "/home/ref-argo/gdac" if not os.uname()[0] == 'Darwin' else "~/data/ARGO"
    THIS_PROFILE = store(host=host).search_wmo_cyc(WMO, CYC).to_dataframe()
    THIS_DATE = pd.to_datetime(THIS_PROFILE['date'].values[0])
    CENTER = [THIS_PROFILE['longitude'].values[0], THIS_PROFILE['latitude'].values[0]]
    if not args.json:
        puts("\nProfiles to work with:")
        puts(THIS_PROFILE.to_string(max_colwidth=15), color=COLORS.green)
        if THIS_PROFILE.shape[0] == 1:
            puts('\nReal-case scenario: True position unknown !', color=COLORS.yellow)
        else:
            puts('\nEvaluation scenario: historical position known', color=COLORS.yellow)

    # Load real float configuration at the previous cycle:
    if not args.json:
        puts("\nLoading float configuration...")
    try:
        CFG = FloatConfiguration([WMO, CYC[0]])
    except:
        if not args.json:
            puts("Can't load this profile config, falling back on default values", color=COLORS.red)
        CFG = FloatConfiguration('default')

    if args.cfg_parking_depth is not None:
        puts("parking_depth=%i is overwritten with %i" % (CFG.mission['parking_depth'],
                                                          float(args.cfg_parking_depth)))
        CFG.update('parking_depth', float(args.cfg_parking_depth))

    if args.cfg_cycle_duration is not None:
        puts("cycle_duration=%i is overwritten with %i" % (CFG.mission['cycle_duration'],
                                                          float(args.cfg_cycle_duration)))
        CFG.update('cycle_duration', float(args.cfg_cycle_duration))

    # Save virtual float configuration on file:
    CFG.to_json(os.path.join(WORKDIR, "floats_configuration_%s.json" % get_sim_suffix(args, CFG)))

    if not args.json:
        puts("\n".join(["\t%s" % line for line in CFG.__repr__().split("\n")]), color=COLORS.green)

    # Get the cycling frequency (in days):
    # dt = pd.to_datetime(THIS_PROFILE['date'].values[1]) - pd.to_datetime(THIS_PROFILE['date'].values[0])
    # CYCLING_FREQUENCY = int(np.round(dt.days + dt.seconds / 86400))
    CYCLING_FREQUENCY = int(np.round(CFG.mission['cycle_duration'])/24)

    # Define domain to load velocity for, and get it:
    width = 5 + np.abs(np.ceil(THIS_PROFILE['longitude'].values[-1] - CENTER[0]))
    height = 5 + np.abs(np.ceil(THIS_PROFILE['latitude'].values[-1] - CENTER[1]))
    # lonc, latc = CENTER[0], CENTER[1],
    VBOX = [CENTER[0] - width / 2, CENTER[0] + width / 2, CENTER[1] - height / 2, CENTER[1] + height / 2]
    if not args.json:
        puts("\nLoading %s velocity field to cover %i days..." % (VEL_NAME, CYCLING_FREQUENCY+1))
    ds_vel, velocity_file = get_velocity_field(VBOX, THIS_DATE,
                                           n_days=CYCLING_FREQUENCY+1,
                                           output=WORKDIR,
                                           dataset=VEL_NAME)
    VEL = VelocityField(model='GLORYS12V1' if VEL_NAME == 'GLORYS' else VEL_NAME, src=ds_vel)
    if not args.json:
        puts("\tLoaded velocity field from %s to %s" %
             (pd.to_datetime(ds_vel['time'][0].values).strftime("%Y-%m-%dT%H:%M:%S"),
              pd.to_datetime(ds_vel['time'][-1].values).strftime("%Y-%m-%dT%H:%M:%S")), color=COLORS.green)
    fig, ax = figure_velocity(VBOX, VEL, VEL_NAME, THIS_PROFILE, WMO, CYC, save_figure=args.save_figure, workdir=WORKDIR)

    # VirtualFleet, get a deployment plan:
    if not args.json:
        puts("\nVirtualFleet, get a deployment plan...")
    DF_PLAN = setup_deployment_plan(CENTER, THIS_DATE, nfloats=args.nfloats)
    PLAN = {'lon': DF_PLAN['longitude'],
            'lat': DF_PLAN['latitude'],
            'time': np.array([np.datetime64(t) for t in DF_PLAN['date'].dt.strftime('%Y-%m-%d %H:%M').array]),
            }
    if not args.json:
        puts("\t%i virtual floats to deploy" % DF_PLAN.shape[0], color=COLORS.green)

    # VirtualFleet, set-up the fleet:
    if not args.json:
        puts("\nVirtualFleet, set-up the fleet...")
    VFleet = VirtualFleet(plan=PLAN,
                          fieldset=VEL,
                          mission=CFG)

    # VirtualFleet, execute the simulation:
    if not args.json:
        puts("\nVirtualFleet, execute the simulation...")

    # Remove traj file if exists:
    output_path = os.path.join(WORKDIR, 'trajectories_%s.zarr' % get_sim_suffix(args, CFG))
    if args.save_sim and os.path.exists(output_path):
        shutil.rmtree(output_path)

    VFleet.simulate(duration=timedelta(hours=CYCLING_FREQUENCY*24+1),
                    step=timedelta(minutes=5),
                    record=timedelta(minutes=30),
                    output=args.save_sim,
                    output_folder=WORKDIR,
                    output_file='trajectories_%s.zarr' % get_sim_suffix(args, CFG),
                    verbose_progress=not args.json,
                    )

    # VirtualFleet, get simulated profiles index:
    if not args.json:
        puts("\nVirtualFleet, extract simulated profiles index...")
    # ds_traj = xr.open_dataset(VFleet.output)
    # DF_SIM = simu2index_legacy(DF_PLAN, ds_traj)
    # DF_SIM = ds_simu2index(ds_traj)
    try:
        DF_SIM = get_index(VFleet, VEL, DF_PLAN)
    except ValueError:
        ds_traj = xr.open_zarr(VFleet.output)
        DF_SIM = ds_simu2index(ds_traj)
    DF_SIM = postprocess_index(DF_SIM, THIS_PROFILE)
    if not args.json:
        puts(DF_SIM.head().to_string(), color=COLORS.green)
    fig, ax = figure_positions(args, VEL, DF_SIM, DF_PLAN, THIS_PROFILE, CFG, WMO, CYC, VEL_NAME,
                               dd=1, save_figure=args.save_figure, workdir=WORKDIR)

    # Recovery, make predictions based on simulated profile density:
    results = predict_position(args, WORKDIR, WMO, CYC, CFG, VEL, VEL_NAME, DF_SIM, DF_PLAN, THIS_PROFILE,
                               save_figure=args.save_figure, quiet=~args.json)
    results['profile_to_predict'] = {'wmo': WMO,
                          'cycle_number': CYC[-1],
                          'url_float': argoplot.dashboard(WMO, url_only=True),
                          'url_profile': None,
                          'location': {'longitude': {'value': None,
                                                     'unit': 'degree East'},
                                       'latitude': {'value': None,
                                                    'unit': 'degree North'},
                                       'time': {'value': None}}
                                     }
    if THIS_PROFILE.shape[0] > 1:
        results['profile_to_predict']['url_profile'] = argoplot.dashboard(WMO, CYC[-1], url_only=True)
        results['profile_to_predict']['location']['longitude']['value'] = THIS_PROFILE['longitude'].values[-1]
        results['profile_to_predict']['location']['latitude']['value'] = THIS_PROFILE['latitude'].values[-1]
        results['profile_to_predict']['location']['time']['value'] = THIS_PROFILE['date'].values[-1]

    results['previous_profile'] = {'wmo': WMO,
                          'cycle_number': CYC[0],
                          'url_float': argoplot.dashboard(WMO, url_only=True),
                          'url_profile': argoplot.dashboard(WMO, CYC[0], url_only=True),
                          'location': {'longitude': {'value': CENTER[0],
                                                     'unit': 'degree East'},
                                       'latitude': {'value': CENTER[1],
                                                    'unit': 'degree North'},
                                       'time': {'value': THIS_DATE}}
                                   }
    results = analyse_pairwise_distances(args, CFG, results)

    execution_end = time.time()
    process_end = time.process_time()
    computation = {
        'Date': pd.to_datetime('now', utc=True),
        'Wall-time': pd.Timedelta(execution_end - execution_start, 's'),
        'CPU-time': pd.Timedelta(process_end - process_start, 's'),
        'system': getSystemInfo()
    }
    results['meta'] = {'Velocity field': VEL_NAME,
                       'Nfloats': args.nfloats,
                       'Computation': computation,
                       'VFloats_config': CFG.to_json(),
                       }

    if not args.json:
        puts("\nPredictions:")
    results_js = json.dumps(results, indent=4, sort_keys=True, default=str)

    # with open(os.path.join(WORKDIR, 'prediction.json'), 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=4, default=str, sort_keys=True)

    with open(os.path.join(WORKDIR, 'prediction_%s.json' % get_sim_suffix(args, CFG)), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4, default=str, sort_keys=True)

    if not args.json:
        puts(results_js, color=COLORS.green)

        puts("\nCheck results at:")
        puts("\t%s" % WORKDIR, color=COLORS.green)

    if args.save_figure:
        # Restore Matplotlib backend
        matplotlib.use(mplbackend)

    return results_js

if __name__ == '__main__':
    # Read mandatory arguments from the command line
    ARGS = setup_args().parse_args()
    js = predictor(ARGS)
    if ARGS.json:
        sys.stdout.write(js)

    # Exit gracefully
    sys.exit(0)
