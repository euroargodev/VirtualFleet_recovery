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
    color=None
    bold=False
    file=sys.stdout
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
    Calculate the great circle distance (in [km]) between two points
    on the earth (specified in decimal degrees)

    see: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points

    Parameters
    ----------
    lon1
    lat1
    lon2
    lat2

    Returns
    -------
    km
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


def get_glorys_forecast_from_datarmor(a_box, a_start_date, n_days=1):
    """Load Datarmor Global Ocean 1/12° Physics Analysis and Forecast updated Daily

    Fields: daily, from 2020-11-25T12:00 to 'now' + 5 days
    Src: /home/ref-ocean-model-public/multiparameter/physic/global/cmems/global-analysis-forecast-phy-001-024
    Info: https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSISFORECAST_PHY_001_024/INFORMATION

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


class Armor3d:
    """Global Ocean 1/4° Multi Observation Product ARMOR3D

    Product description:
    https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012

    If start_date + n_days <= 2022-12-28:
        Delivers the multi-year reprocessed (REP) weekly data

    otherwise:
        Delivers the near-real-time (NRT) weekly data

    Examples
    --------
    >>> Armor3d([-25, -13, 6.5, 13], pd.to_datetime('20091130', utc=True)).to_xarray()
    >>> Armor3d([-25, -13, 6.5, 13], pd.to_datetime('20231121', utc=True), n_days=10).to_xarray()

    """

    def __init__(self, box, start_date, n_days=1, max_depth=2500):
        """
        Parameters
        ----------
        box: list(float)
            Define domain to load: [lon_min, lon_max, lat_min, lat_max]
        start_date: :class:`pandas.Timestamp`
            Starting date of the time series to load. Since ARMOR3D is weekly, the effective starting
            date will be the first weekly period including the user-defined ``start_date``
        n_days: int (default=1)
            Number of days to load data for.
        max_depth: float (default=2500)
            Maximum depth levels to load data for.
        """
        self.box = box
        self.start_date = start_date
        self.n_days = n_days
        self.max_depth = max_depth

        dt = pd.Timedelta(n_days, 'D') if n_days > 1 else pd.Timedelta(0, 'D')
        if start_date + dt <= pd.to_datetime('2022-12-28', utc=True):
            self._loader = self._get_rep
            self.dataset_id = "dataset-armor-3d-rep-weekly"
            self.time_axis = pd.Series(pd.date_range('19930106', '20221228', freq='7D').tz_localize("UTC"))
        else:
            self._loader = self._get_nrt
            self.dataset_id = "dataset-armor-3d-nrt-weekly"
            self.time_axis = pd.Series(
                pd.date_range('20190102', pd.to_datetime('now', utc=True).strftime("%Y%m%d"), freq='7D').tz_localize(
                    "UTC")[0:-1])

        if start_date < self.time_axis.iloc[0]:
            raise ValueError('Date out of bounds')
        elif start_date + dt > self.time_axis.iloc[-1]:
            raise ValueError('Date out of bounds, %s > %s' % (
                start_date + dt, self.time_axis.iloc[-1]))

    def _get_this(self, dataset_id):
        start_date = self.time_axis[self.time_axis <= self.start_date].iloc[-1]
        if self.n_days == 1:
            end_date = start_date
        else:
            end_date = \
            self.time_axis[self.time_axis <= self.start_date + (self.n_days + 1) * pd.Timedelta(1, 'D')].iloc[-1]

        ds = copernicusmarine.open_dataset(
            dataset_id=dataset_id,
            minimum_longitude=self.box[0],
            maximum_longitude=self.box[1],
            minimum_latitude=self.box[2],
            maximum_latitude=self.box[3],
            maximum_depth=self.max_depth,
            start_datetime=start_date.strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=end_date.strftime("%Y-%m-%dT%H:%M:%S"),
            variables=['ugo', 'vgo']
        )
        return ds

    def _get_rep(self):
        """multi-year reprocessed (REP) weekly data

        Returns
        -------
        :class:xarray.dataset
        """
        return self._get_this(self.dataset_id)

    def _get_nrt(self):
        """near-real-time (NRT) weekly data

        Returns
        -------
        :class:xarray.dataset
        """
        return self._get_this(self.dataset_id)

    def to_xarray(self):
        """Load and return data as a :class:`xarray.dataset`

        Returns
        -------
        :class:xarray.dataset
        """
        return self._loader()

    def __repr__(self):
        summary = ["<CopernicusMarineData.Loader><Armor3D>"]
        summary.append("dataset_id: %s" % self.dataset_id)
        summary.append("First day: %s" % self.start_date)
        summary.append("N days: %s" % self.n_days)
        summary.append("Domain: %s" % self.box)
        summary.append("Max depth (m): %s" % self.max_depth)
        return "\n".join(summary)


class Glorys:
    """Global Ocean 1/12° Physics Re-Analysis or Forecast

    If start_date + n_days <= 2021-01-09:
        Delivers the multi-year reprocessed (REP) daily data
        https://resources.marine.copernicus.eu/product-detail/GLOBAL_MULTIYEAR_PHY_001_030

    otherwise:
        Delivers the near-real-time (NRT) Analysis and Forecast daily data
        https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSISFORECAST_PHY_001_024

    Examples
    --------
    >>> Glorys([-25, -13, 6.5, 13], pd.to_datetime('20091130', utc=True)).to_xarray()
    >>> Glorys([-25, -13, 6.5, 13], pd.to_datetime('20231121', utc=True), n_days=10).to_xarray()

    """

    def __init__(self, box, start_date, n_days=1, max_depth=2500):
        """
        Parameters
        ----------
        box: list(float)
            Define domain to load: [lon_min, lon_max, lat_min, lat_max]
        start_date: :class:`pandas.Timestamp`
            Starting date of the time series to load.
        n_days: int (default=1)
            Number of days to load data for.
        max_depth: float (default=2500)
            Maximum depth levels to load data for.
        """
        self.box = box
        self.start_date = start_date
        self.n_days = n_days
        self.max_depth = max_depth

        dt = pd.Timedelta(n_days, 'D') if n_days > 1 else pd.Timedelta(0, 'D')
        if start_date + dt <= pd.to_datetime('2021-01-09', utc=True):
            self._loader = self._get_reanalysis
            self.dataset_id = "cmems_mod_glo_phy_my_0.083_P1D-m"
        else:
            self._loader = self._get_forecast
            self.dataset_id = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"

    def _get_this(self, dataset_id, dates):
        ds = copernicusmarine.open_dataset(
            dataset_id=dataset_id,
            minimum_longitude=self.box[0],
            maximum_longitude=self.box[1],
            minimum_latitude=self.box[2],
            maximum_latitude=self.box[3],
            maximum_depth=self.max_depth,
            start_datetime=dates[0].strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=dates[1].strftime("%Y-%m-%dT%H:%M:%S"),
            variables=['uo', 'vo']
        )
        return ds

    def _get_forecast(self):
        """
        Returns
        -------
        :class:`xarray.dataset`
        """
        start_date = self.start_date
        if self.n_days == 1:
            end_date = start_date
        else:
            end_date = start_date + pd.Timedelta(self.n_days - 1, 'D')
        return self._get_this(self.dataset_id, [start_date, end_date])

    def _get_reanalysis(self):
        """
        Returns
        -------
        :class:`xarray.dataset`
        """
        start_date = self.start_date
        if self.n_days == 1:
            end_date = start_date
        else:
            end_date = self.start_date + pd.Timedelta(self.n_days - 1, 'D')
        return self._get_this(self.dataset_id, [start_date, end_date])

    def to_xarray(self):
        """ Load and return data as a :class:`xarray.dataset`
        Returns
        -------
        :class:`xarray.dataset`
        """
        return self._loader()

    def __repr__(self):
        summary = ["<CopernicusMarineData.Loader><Glorys>"]
        summary.append("dataset_id: %s" % self.dataset_id)
        summary.append("First day: %s" % self.start_date)
        summary.append("N days: %s" % self.n_days)
        summary.append("Domain: %s" % self.box)
        summary.append("Max depth (m): %s" % self.max_depth)
        return "\n".join(summary)


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
    def get_velocity_filename(dataset, n_days):
        download_date = pd.to_datetime('now', utc='now').strftime("%Y%m%d")
        fname = os.path.join(output, 'velocity_%s_%idays_%s.nc' % (dataset, n_days, download_date))
        return fname

    velocity_file = get_velocity_filename(dataset, n_days)
    if not os.path.exists(velocity_file):
        # Define Data loader:
        loader = Armor3d if dataset == 'ARMOR3D' else Glorys
        loader = loader(a_box, a_date, n_days=n_days)
        puts(str(loader), color=COLORS.magenta)

        # Load data from Copernicus Marine Data store:
        ds = loader.to_xarray()

        # Save on file for later re-used:
        ds.to_netcdf(velocity_file)
    else:
        ds = xr.open_dataset(velocity_file)

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
    txt = "VFloat configuration: (Parking depth: %i [db], Cycle duration: %i [hours], Profile depth: %i [db])" % (
        a_cfg.mission['parking_depth'],
        a_cfg.mission['cycle_duration'],
        a_cfg.mission['profile_depth'],
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

        ax[ix].plot(df_sim['deploy_lon'], df_sim['deploy_lat'], '.',
                    markersize=3, color='grey', alpha=0.1, markeredgecolor=None, zorder=0)
        if ix == 0:
            title = 'Velocity field at %0.2fm and deployment plan' % cfg.mission['parking_depth']
            v.set_alpha(1)
            # v.set_color('black')
        elif ix == 1:
            x, y, c = df_sim['longitude'], df_sim['latitude'], df_sim['cyc']
            title = 'Final float positions'
            # sc = ax[ix].plot(x, y, '.', markersize=3, color='cyan', alpha=0.9, markeredgecolor=None)
            sc = ax[ix].scatter(x, y, c=c, s=3, alpha=0.9, edgecolors=None)
        elif ix == 2:
            x, y, c = df_sim['rel_lon'], df_sim['rel_lat'], df_sim['cyc']
            title = 'Final floats position relative to last float position'
            # sc = ax[ix].plot(x, y, '.', markersize=3, color='cyan', alpha=0.9, markeredgecolor=None)
            sc = ax[ix].scatter(x, y, c=c, s=3, alpha=0.9, edgecolors=None)

        ax[ix] = map_add_profiles(ax[ix], this_profile)
        ax[ix].set_title(title)

    fig.suptitle("VirtualFleet recovery prediction for WMO %i: starting from cycle %i, predicting cycle %s\n%s" %
                 (wmo, cyc[0], cyc[1:], get_cfg_str(cfg)), fontsize=15)
    plt.tight_layout()
    if save_figure:
        save_figurefile(fig, "vfrecov_positions_%s" % get_sim_suffix(this_args, cfg), workdir)
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
    tim = tim.round(freq='5min')

    #
    df = pd.DataFrame(
        [tim, lat, lon, np.arange(0, nfloats) + 9000000, np.full_like(lon, 0), ['VF' for l in lon], ['?' for l in lon]],
        index=['date', 'latitude', 'longitude', 'wmo', 'cycle_number', 'institution_code', 'file']).T
    df['date'] = pd.to_datetime(df['date'])

    return df


class Trajectories:
    """Trajectory file manager for VFrecovery

    Examples:
    ---------
    T = Trajectories(traj_zarr_file)
    T.n_floats
    T.sim_cycles
    df = T.to_index()
    df = T.get_index().add_distances()
    jsdata, fig, ax = T.analyse_pairwise_distances(cycle=1, show_plot=True)
    """

    def __init__(self, zfile):
        self.zarr_file = zfile
        self.obj = xr.open_zarr(zfile)
        self._index = None

    @property
    def n_floats(self):
        # len(self.obj['trajectory'])
        return self.obj['trajectory'].shape[0]

    @property
    def sim_cycles(self):
        """Return list of cycles simulated"""
        cycs = np.unique(self.obj['cycle_number'])
        last_obs_phase = \
        self.obj.where(self.obj['cycle_number'] == cycs[-1])['cycle_phase'].isel(trajectory=0).isel(obs=-1).values[
            np.newaxis][0]
        if last_obs_phase < 3:
            cycs = cycs[0:-1]
        return cycs

    def __repr__(self):
        summary = ["<VRecovery.Trajectories>"]
        summary.append("Swarm size: %i floats" % self.n_floats)
        start_date = pd.to_datetime(self.obj['time'].isel(trajectory=0, obs=0).values)
        end_date = pd.to_datetime(self.obj['time'].isel(trajectory=0, obs=-1).values)
        summary.append("Simulation length: %s, from %s to %s" % (
        pd.Timedelta(end_date - start_date, 'd'), start_date.strftime("%Y/%m/%d"), end_date.strftime("%Y/%m/%d")))
        return "\n".join(summary)

    def to_index_par(self) -> pd.DataFrame:
        # Deployment loc:
        deploy_lon, deploy_lat = self.obj.isel(obs=0)['lon'].values, self.obj.isel(obs=0)['lat'].values

        def worker(ds, cyc, x0, y0):
            mask = np.logical_and((ds['cycle_number'] == cyc).compute(),
                                  (ds['cycle_phase'] >= 3).compute())
            this_cyc = ds.where(mask, drop=True)
            if len(this_cyc['time']) > 0:
                data = {
                    'date': this_cyc.isel(obs=-1)['time'].values,
                    'latitude': this_cyc.isel(obs=-1)['lat'].values,
                    'longitude': this_cyc.isel(obs=-1)['lon'].values,
                    'wmo': 9000000 + this_cyc.isel(obs=-1)['trajectory'].values,
                    'cyc': cyc,
                    # 'cycle_phase': this_cyc.isel(obs=-1)['cycle_phase'].values,
                    'deploy_lon': x0,
                    'deploy_lat': y0,
                }
                return pd.DataFrame(data)
            else:
                return None

        cycles = np.unique(self.obj['cycle_number'])
        rows = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {
                executor.submit(
                    worker,
                    self.obj,
                    cyc,
                    deploy_lon,
                    deploy_lat
                ): cyc
                for cyc in cycles
            }
            futures = concurrent.futures.as_completed(future_to_url)
            for future in futures:
                data = None
                try:
                    data = future.result()
                except Exception:
                    raise
                finally:
                    rows.append(data)

        rows = [r for r in rows if r is not None]
        df = pd.concat(rows).reset_index()
        df['wmo'] = df['wmo'].astype(int)
        df['cyc'] = df['cyc'].astype(int)
        # df['cycle_phase'] = df['cycle_phase'].astype(int)
        self._index = df

        return self._index

    def to_index(self) -> pd.DataFrame:
        """Compute and return index (profile dataframe from trajectory dataset)

        Create a Profile index :class:`pandas.dataframe` with columns: [data, latitude ,longitude, wmo, cyc, deploy_lon, deploy_lat]
        from a trajectory :class:`xarray.dataset`.

        There is one dataframe row for each dataset trajectory cycle.

        We use the last trajectory point of given cycle number (with cycle phase >= 3) to identify a profile location.

        If they are N trajectories simulating C cycles, there will be about a maximum of N*C rows in the dataframe.

        Returns
        -------
        :class:`pandas.dataframe`
        """
        if self._index is None:

            # Deployment loc:
            deploy_lon, deploy_lat = self.obj.isel(obs=0)['lon'].values, self.obj.isel(obs=0)['lat'].values

            def worker(ds, cyc, x0, y0):
                mask = np.logical_and((ds['cycle_number'] == cyc).compute(),
                                      (ds['cycle_phase'] >= 3).compute())
                this_cyc = ds.where(mask, drop=True)
                if len(this_cyc['time']) > 0:
                    data = {
                        'date': this_cyc.isel(obs=-1)['time'].values,
                        'latitude': this_cyc.isel(obs=-1)['lat'].values,
                        'longitude': this_cyc.isel(obs=-1)['lon'].values,
                        'wmo': 9000000 + this_cyc.isel(obs=-1)['trajectory'].values,
                        'cyc': cyc,
                        # 'cycle_phase': this_cyc.isel(obs=-1)['cycle_phase'].values,
                        'deploy_lon': x0,
                        'deploy_lat': y0,
                    }
                    return pd.DataFrame(data)
                else:
                    return None

            cycles = np.unique(self.obj['cycle_number'])
            rows = []
            for cyc in cycles:
                df = worker(self.obj, cyc, deploy_lon, deploy_lat)
                rows.append(df)
            rows = [r for r in rows if r is not None]
            df = pd.concat(rows).reset_index()
            df['wmo'] = df['wmo'].astype(int)
            df['cyc'] = df['cyc'].astype(int)
            # df['cycle_phase'] = df['cycle_phase'].astype(int)
            self._index = df

        return self._index

    def get_index(self):
        """Compute index and return self"""
        self.to_index()
        return self

    def add_distances(self, origin: None) -> pd.DataFrame:
        """Compute profiles distance to some origin

        Returns
        -------
        :class:`pandas.dataframe`
        """

        # Compute distance between the predicted profile and the initial profile location from the deployment plan
        # We assume that virtual floats are sequentially taken from the deployment plan
        # Since distances are very short, we compute a simple rectangular distance

        # Observed cycles:
        # obs_cyc = np.unique(this_profile['cyc'])

        # Simulated cycles:
        # sim_cyc = np.unique(this_df['cyc'])

        df = self._index

        x2, y2 = origin  # real float initial position
        df['distance'] = np.nan
        df['rel_lon'] = np.nan
        df['rel_lat'] = np.nan
        df['distance_origin'] = np.nan

        def worker(row):
            # Simulation profile coordinates:
            x0, y0 = row['deploy_lon'], row['deploy_lat']  # virtual float initial position
            x1, y1 = row['longitude'], row['latitude']  # virtual float position

            # Distance between each pair of cycles of virtual floats:
            dist = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
            row['distance'] = dist

            # Shift between each pair of cycles:
            dx, dy = x1 - x0, y1 - y0
            # Get a relative displacement from real float initial position:
            row['rel_lon'] = x2 + dx
            row['rel_lat'] = y2 + dy

            # Distance between the predicted profile and the observed initial profile
            dist = np.sqrt((y2 - y0) ** 2 + (x2 - x0) ** 2)
            row['distance_origin'] = dist

            return row

        df = df.apply(worker, axis=1)
        self._index = df

        return self._index

    def analyse_pairwise_distances(self,
                                   cycle: int = 1,
                                   show_plot: bool = True,
                                   save_figure: bool = False,
                                   workdir: str = '.',
                                   sim_suffix = None,
                                   this_cfg = None,
                                   this_args: dict = None):

        def get_hist_and_peaks(this_d):
            x = this_d.flatten()
            x = x[~np.isnan(x)]
            x = x[:, np.newaxis]
            hist, bin_edges = np.histogram(x, bins=100, density=1)
            # dh = np.diff(bin_edges[0:2])
            peaks, _ = find_peaks(hist / np.max(hist), height=.4, distance=20)
            return {'pdf': hist, 'bins': bin_edges[0:-1], 'Npeaks': len(peaks)}

        # Squeeze traj file to the first predicted cycle (sim can have more than 1 cycle)
        ds = self.obj.where((self.obj['cycle_number'] == cycle).compute(), drop=True)
        ds = ds.compute()

        # Compute trajectories relative to the single/only real float initial position:
        lon0, lat0 = self.obj.isel(obs=0)['lon'].values[0], self.obj.isel(obs=0)['lat'].values[0]
        lon, lat = ds['lon'].values, ds['lat'].values
        ds['lonc'] = xr.DataArray(lon - np.broadcast_to(lon[:, 0][:, np.newaxis], lon.shape) + lon0,
                                  dims=['trajectory', 'obs'])
        ds['latc'] = xr.DataArray(lat - np.broadcast_to(lat[:, 0][:, np.newaxis], lat.shape) + lat0,
                                  dims=['trajectory', 'obs'])

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

        # Store metrics in a dict:
        prediction_metrics = {}

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

        # Figure:
        if show_plot:
            backend = matplotlib.get_backend()
            if this_args is not None and this_args.json:
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

            if this_args is not None:
                line0 = "VirtualFleet recovery swarm simulation for WMO %i, starting from cycle %i, predicting cycle %i\n%s" % \
                        (this_args.wmo, this_args.cyc[0] - 1, this_args.cyc[0], get_cfg_str(this_cfg))
                line1 = "Simulation made with %s and %i virtual floats" % (this_args.velocity, this_args.nfloats)
            else:
                line0 = "VirtualFleet recovery swarm simulation for cycle %i" % cycle
                line1 = "Simulation made with %i virtual floats" % (self.n_floats)

            fig.suptitle("%s\n%s" % (line0, line1), fontsize=15)
            plt.tight_layout()

            if save_figure:
                if sim_suffix is not None:
                    filename = 'vfrecov_metrics01_%s_cyc%i' % (sim_suffix, cycle)
                else:
                    filename = 'vfrecov_metrics01_cyc%i' % (cycle)
                save_figurefile(fig, filename, workdir)

            if this_args is not None and this_args.json:
                matplotlib.use(backend)

        if show_plot:
            return prediction_metrics, fig, ax
        else:
            return prediction_metrics


class SimPredictor_0:
    """

    Examples
    --------
    T = Trajectories(traj_zarr_file)
    df = T.get_index().add_distances()

    SP = SimPredictor(df)
    SP.fit_predict()
    SP.add_metrics(VFvelocity)
    SP.bbox()
    SP.plot_predictions(VFvelocity)
    SP.plan
    SP.n_cycles
    SP.trajectory
    SP.prediction
    """

    def __init__(self, df_sim: pd.DataFrame, df_obs: pd.DataFrame):
        self.swarm = df_sim
        self.obs = df_obs
        # self.set_weights()
        self.WMO = np.unique(df_obs['wmo'])[0]
        self._json = None

    def __repr__(self):
        summary = ["<VFRecovery.Predictor>"]
        summary.append("Simulation target: %i / %i" % (self.WMO, self.sim_cycles[0]))
        summary.append("Swarm size: %i floats" % len(np.unique(self.swarm['wmo'])))
        summary.append("Number of simulated cycles: %i profile(s) for cycle number(s): [%s]" % (
        self.n_cycles, ",".join([str(c) for c in self.sim_cycles])))
        summary.append("Observed reference: %i profile(s) for cycle number(s): [%s]" % (
        self.obs.shape[0], ",".join([str(c) for c in self.obs_cycles])))
        return "\n".join(summary)

    @property
    def n_cycles(self):
        """Number of simulated cycles"""
        return len(np.unique(self.swarm['cyc']))
        # return len(self.sim_cycles)

    @property
    def obs_cycles(self):
        """Observed cycle numbers"""
        return np.unique(self.obs['cyc'])

    @property
    def sim_cycles(self):
        """Simulated cycle numbers"""
        return self.obs_cycles[0] + 1 + range(self.n_cycles)

    @property
    def plan(self) -> pd.DataFrame:
        if not hasattr(self, '_plan'):
            df_plan = self.swarm[self.swarm['cyc'] == 1][['date', 'deploy_lon', 'deploy_lat']]
            df_plan = df_plan.rename(columns={'deploy_lon': 'longitude', 'deploy_lat': 'latitude'})
            self._plan = df_plan
        return self._plan

    @property
    def trajectory(self):
        """Return the predicted trajectory as a simple :class:`np.array`

        First row is longitude, 2nd is latitude and 3rd is date of simulated profiles

        Return
        ------
        :class:`np.array`

        """
        if self._json is None:
            raise ValueError("Please call `fit_predict` first")

        traj_prediction = np.array([self.obs['longitude'].values[0],
                                    self.obs['latitude'].values[0],
                                    self.obs['date'].values[0]])[
            np.newaxis]  # Starting point where swarm was deployed
        for cyc in self._json['predictions'].keys():
            xpred = self._json['predictions'][cyc]['location']['longitude']
            ypred = self._json['predictions'][cyc]['location']['latitude']
            tpred = pd.to_datetime(self._json['predictions'][cyc]['location']['time'])
            traj_prediction = np.concatenate((traj_prediction,
                                              np.array([xpred, ypred, tpred])[np.newaxis]),
                                             axis=0)
        return traj_prediction

    @property
    def predictions(self):
        if self._json is None:
            raise ValueError("Please call `fit_predict` first")
        return self._json

    def bbox(self, s: float = 1) -> list:
        """Get a bounding box for maps

        Parameters
        ----------
        s: float, default:1

        Returns
        -------
        list
        """
        df_sim = self.swarm
        df_obs = self.obs

        box = [np.min([df_sim['deploy_lon'].min(),
                       df_sim['longitude'].min(),
                       df_sim['rel_lon'].min(),
                       df_obs['longitude'].min()]),
               np.max([df_sim['deploy_lon'].max(),
                       df_sim['longitude'].max(),
                       df_sim['rel_lon'].max(),
                       df_obs['longitude'].max()]),
               np.min([df_sim['deploy_lat'].min(),
                       df_sim['latitude'].min(),
                       df_sim['rel_lat'].min(),
                       df_obs['latitude'].min()]),
               np.max([df_sim['deploy_lat'].max(),
                       df_sim['latitude'].max(),
                       df_sim['rel_lat'].max(),
                       df_obs['latitude'].max()])]
        rx, ry = box[1] - box[0], box[3] - box[2]
        r = np.min([rx, ry])
        ebox = [box[0] - s * r, box[1] + s * r, box[2] - s * r, box[3] + s * r]

        return ebox

class SimPredictor_1(SimPredictor_0):

    def set_weights(self, scale: float = 20):
        """Compute weights for predictions

        Add weights column to swarm :class:`pandas.DataFrame` as a gaussian distance
        with a std based on the size of the deployment domain

        Parameters
        ----------
        scale: float (default=20.)
        """
        rx, ry = self.plan['longitude'].max() - self.plan['longitude'].min(), \
                 self.plan['latitude'].max() - self.plan['latitude'].min()
        r = np.min([rx, ry])  # Minimal size of the deployment domain
        weights = np.exp(-(self.swarm['distance_origin'] ** 2) / (r / scale))
        weights[np.isnan(weights)] = 0
        self.swarm['weights'] = weights
        return self

    def fit_predict(self, weights_scale: float = 20.) -> dict:
        """Predict profile positions from simulated float swarm

        Prediction is based on a :class:`klearn.neighbors._kde.KernelDensity` estimate of the N_FLOATS
        simulated, weighted by their deployment distance to the observed previous cycle position.

        Parameters
        ----------
        weights_scale: float (default=20)
            Scale (in deg) to use to weight the deployment distance to the observed previous cycle position

        Returns
        -------
        dict
        """

        def blank_prediction() -> dict:
            return {'location': {
                        'longitude': None,
                        'latitude': None,
                        'time': None},
                    'cycle_number': None,
                    'wmo': int(self.WMO),
                    }

        # Compute weights of the swarm float profiles locations
        self.set_weights(scale=weights_scale)

        self._prediction_data = {'weights_scale': weights_scale, 'cyc': {}}

        cycles = np.unique(self.swarm['cyc']).astype(int)  # 1, 2, ...
        recovery_predictions = {}
        for icyc, this_sim_cyc in enumerate(cycles):
            this_cyc_df = self.swarm[self.swarm['cyc'] == this_sim_cyc]
            weights = this_cyc_df['weights']
            x, y = this_cyc_df['rel_lon'], this_cyc_df['rel_lat']

            w = weights / np.max(np.abs(weights), axis=0)
            X = np.array([x, y]).T
            kde = KernelDensity(kernel='gaussian', bandwidth=0.15).fit(X, sample_weight=w)

            xg, yg = (np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100),
                      np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100))
            xg, yg = np.meshgrid(xg, yg)
            Xg = np.array([xg.flatten(), yg.flatten(), ]).T
            llh = kde.score_samples(Xg)
            xpred = Xg[np.argmax(llh), 0]
            ypred = Xg[np.argmax(llh), 1]
            tpred = this_cyc_df['date'].mean()

            # Store results
            recovery = blank_prediction()
            recovery['location']['longitude'] = xpred
            recovery['location']['latitude'] = ypred
            recovery['location']['time'] = tpred.isoformat()
            recovery['cycle_number'] = int(self.sim_cycles[icyc])
            recovery['virtual_cycle_number'] = int(self.sim_cycles[icyc])
            recovery_predictions.update({int(this_sim_cyc): recovery})

            #
            self._prediction_data['cyc'].update({this_sim_cyc: {'weights': this_cyc_df['weights']}})

        # Store results internally
        self._json = {'predictions': recovery_predictions}

        # Add more stuff to internal storage:
        self._predict_errors()
        self._add_ref()
        self.add_metrics()

        #
        return self


class SimPredictor_2(SimPredictor_1):

    def _predict_errors(self) -> dict:
        """Compute error metrics for the predicted positions

        This is for past cycles, for which we have observed positions of the predicted profiles

        This adds more keys to self._json['predictions'] created by the fit_predict method

        Returns
        -------
        dict
        """

        def blank_error():
            return {'distance': {'value': None,
                                 'unit': 'km'},
                    'bearing': {'value': None,
                                'unit': 'degree'},
                    'time': {'value': None,
                             'unit': 'hour'}
                    }

        cyc0 = self.obs_cycles[0]
        if self._json is None:
            raise ValueError("Please call `fit_predict` first")
        recovery_predictions = self._json['predictions']

        for sim_c in recovery_predictions.keys():
            this_prediction = recovery_predictions[sim_c]
            if sim_c + cyc0 in self.obs_cycles:
                error = blank_error()

                this_obs_profile = self.obs[self.obs['cyc'] == sim_c + cyc0]
                xobs = this_obs_profile['longitude'].iloc[0]
                yobs = this_obs_profile['latitude'].iloc[0]
                tobs = this_obs_profile['date'].iloc[0]

                prev_obs_profile = self.obs[self.obs['cyc'] == sim_c + cyc0 - 1]
                xobs0 = prev_obs_profile['longitude'].iloc[0]
                yobs0 = prev_obs_profile['latitude'].iloc[0]

                xpred = this_prediction['location']['longitude']
                ypred = this_prediction['location']['latitude']
                tpred = pd.to_datetime(this_prediction['location']['time'])

                dd = haversine(xobs, yobs, xpred, ypred)
                error['distance']['value'] = dd

                observed_bearing = bearing(xobs0, yobs0, xobs, yobs)
                sim_bearing = bearing(xobs0, yobs0, xpred, ypred)
                error['bearing']['value'] = sim_bearing - observed_bearing

                dt = pd.Timedelta(tpred - tobs) / np.timedelta64(1, 's')
                # print(tpred, tobs, pd.Timedelta(tpred - tobs))
                error['time']['value'] = dt / 3600  # From seconds to hours

                this_prediction['location_error'] = error
                recovery_predictions.update({sim_c: this_prediction})

        self._json.update({'predictions': recovery_predictions})
        return self

    def _add_ref(self):
        """Add observations data to internal data structure

        This adds more keys to self._json['predictions'] created by the fit_predict method

        """
        if self._json is None:
            raise ValueError("Please call `predict` first")

        # Observed profiles that were simulated:
        profiles_to_predict = []
        for cyc in self.sim_cycles:
            this = {'wmo': int(self.WMO),
                    'cycle_number': int(cyc),
                    'url_float': argoplot.dashboard(self.WMO, url_only=True),
                    'url_profile': "",
                    'location': {'longitude': None,
                                 'latitude': None,
                                 'time': None}
                    }
            if cyc in self.obs_cycles:
                this['url_profile'] = get_ea_profile_page_url(self.WMO, cyc)
                this_df = self.obs[self.obs['cyc'] == cyc]
                this['location']['longitude'] = this_df['longitude'].iloc[0]
                this['location']['latitude'] = this_df['latitude'].iloc[0]
                this['location']['time'] = this_df['date'].iloc[0].isoformat()
            profiles_to_predict.append(this)

        self._json.update({'observations': profiles_to_predict})

        # Observed profile used as initial conditions to the simulation:
        cyc = self.obs_cycles[0]
        this_df = self.obs[self.obs['cyc'] == cyc]
        self._json.update({'initial_profile': {'wmo': int(self.WMO),
                                               'cycle_number': int(cyc),
                                               'url_float': argoplot.dashboard(self.WMO, url_only=True),
                                               'url_profile': get_ea_profile_page_url(self.WMO, cyc),
                                               'location': {'longitude': this_df['longitude'].iloc[0],
                                                            'latitude': this_df['latitude'].iloc[0],
                                                            'time': this_df['date'].iloc[0].isoformat()
                                                            }
                                               }})

        #
        return self

    def add_metrics(self, VFvel=None):
        """Compute more metrics to understand the prediction error

        1. Compute a transit time to cover the distance error
        (assume a 12 kts boat speed with 1 kt = 1.852 km/h)

        1. Compute the possible drift due to the time lag between the predicted profile timing and the expected one

        This adds more keys to self._json['predictions'] created by the fit_predict method

        """
        cyc0 = self.obs_cycles[0]
        if self._json is None:
            raise ValueError("Please call `predict` first")
        recovery_predictions = self._json['predictions']

        for sim_c in recovery_predictions.keys():
            this_prediction = recovery_predictions[sim_c]
            if sim_c + cyc0 in self.obs_cycles and 'location_error' in this_prediction.keys():

                error = this_prediction['location_error']
                metrics = {}

                # Compute a transit time to cover the distance error:
                metrics['transit'] = {'value': None,
                                      'unit': 'hour',
                                      'comment': 'Transit time to cover the distance error '
                                                 '(assume a 12 kts boat speed with 1 kt = 1.852 km/h)'}

                if error['distance']['value'] is not None:
                    metrics['transit']['value'] = pd.Timedelta(error['distance']['value'] / (12 * 1.852),
                                                               'h').seconds / 3600.

                # Compute the possible drift due to the time lag between the predicted profile timing and the expected one:
                if VFvel is not None:
                    xpred = this_prediction['location']['longitude']
                    ypred = this_prediction['location']['latitude']
                    tpred = this_prediction['location']['time']
                    dsc = VFvel.field.interp(
                        {VFvel.dim['lon']: xpred,
                         VFvel.dim['lat']: ypred,
                         VFvel.dim['time']: tpred,
                         VFvel.dim['depth']:
                             VFvel.field[{VFvel.dim['depth']: 0}][VFvel.dim['depth']].values[np.newaxis][0]}
                    )
                    velc = np.sqrt(dsc[VFvel.var['U']] ** 2 + dsc[VFvel.var['V']] ** 2).values[np.newaxis][0]
                    metrics['surface_drift'] = {'value': None,
                                                'unit': 'km',
                                                'surface_currents_speed': None,
                                                'surface_currents_speed_unit': 'm/s',
                                                'comment': 'Drift by surface currents due to the float ascent time error '
                                                           '(difference between simulated profile time and the observed one).'}
                    if error['time']['value'] is not None:
                        metrics['surface_drift']['value'] = (error['time']['value'] * 3600 * velc / 1e3)
                        metrics['surface_drift']['surface_currents_speed'] = velc

                #
                this_prediction['metrics'] = metrics
                recovery_predictions.update({sim_c: this_prediction})

        self._json.update({"predictions": recovery_predictions})
        return self


class SimPredictor_3(SimPredictor_2):

    def plot_predictions(self,
                         VFvel,
                         cfg,
                         sim_suffix='',  # get_sim_suffix(this_args, cfg)
                         s=0.2,
                         alpha=False,
                         save_figure=False,
                         workdir='.',
                         figsize=None,
                         dpi=120,
                         orient='portrait'):
        ebox = self.bbox(s=s)
        pred_traj = self.trajectory

        if orient == 'portrait':
            if self.n_cycles == 1:
                nrows, ncols = 2, 1
                if figsize is None:
                    figsize = (5, 5)
            else:
                nrows, ncols = self.n_cycles, 2
                if figsize is None:
                    figsize = (5, (self.n_cycles-1)*5)
        else:
            if self.n_cycles == 1:
                nrows, ncols = 1, 2
            else:
                nrows, ncols = 2, self.n_cycles
            if figsize is None:
                figsize = (ncols*5, 5)

        def plot_this(this_ax, i_cycle, ip):
            df_sim = self.swarm[self.swarm['cyc'] == i_cycle + 1]
            weights = self._prediction_data['cyc'][i_cycle + 1]['weights'].values
            if self.sim_cycles[i_cycle] in self.obs_cycles:
                this_profile = self.obs[self.obs['cyc'] == self.sim_cycles[i_cycle]]
            else:
                this_profile = None

            xpred = self.predictions['predictions'][i_cycle + 1]['location']['longitude']
            ypred = self.predictions['predictions'][i_cycle + 1]['location']['latitude']

            this_ax.set_extent(ebox)
            this_ax = map_add_features(ax[ix])

            v = VFvel.field.isel(time=0).interp(depth=cfg.mission['parking_depth'])
            v.plot.quiver(x="longitude",
                          y="latitude",
                          u=VFvel.var['U'],
                          v=VFvel.var['V'],
                          ax=this_ax,
                          color='grey',
                          alpha=0.5,
                          scale=5,
                          add_guide=False)

            this_ax.plot(df_sim['deploy_lon'], df_sim['deploy_lat'], '.',
                         markersize=3,
                         color='grey',
                         alpha=0.1,
                         markeredgecolor=None,
                         zorder=0)

            this_ax.plot(pred_traj[:, 0], pred_traj[:, 1], color='k', linewidth=1, marker='+')
            this_ax.plot(xpred, ypred, color='g', marker='+')

            w = weights / np.max(np.abs(weights), axis=0)
            ii = np.argsort(w)
            cmap = plt.cm.cool
            # cmap = plt.cm.Reds

            if ip == 0:
                x, y = df_sim['deploy_lon'], df_sim['deploy_lat']
                title = 'Initial virtual float positions'
                if not alpha:
                    this_ax.scatter(x.iloc[ii], y.iloc[ii], c=w[ii],
                                    marker='o', s=4, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
                else:
                    this_ax.scatter(x.iloc[ii], y.iloc[ii], c=w[ii],
                                    alpha=w[ii],
                                    marker='o', s=4, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
            elif ip == 1:
                x, y = df_sim['longitude'], df_sim['latitude']
                title = 'Final virtual float positions'
                if not alpha:
                    this_ax.scatter(x, y, c=w, marker='o', s=4, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
                else:
                    this_ax.scatter(x, y, c=w, marker='o', s=4, alpha=w, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
            elif ip == 2:
                x, y = df_sim['rel_lon'], df_sim['rel_lat']
                title = 'Final virtual floats positions relative to observed float'
                if not alpha:
                    this_ax.scatter(x.iloc[ii], y.iloc[ii], c=w[ii],
                                    marker='o', s=4, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
                else:
                    this_ax.scatter(x.iloc[ii], y.iloc[ii], c=w[ii],
                                    marker='o', s=4, alpha=w[ii], edgecolor=None, vmin=0, vmax=1, cmap=cmap)

            # Display full trajectory prediction:
            if ip != 0 and this_profile is not None:
                this_ax.arrow(this_profile['longitude'].iloc[0],
                              this_profile['latitude'].iloc[0],
                              xpred - this_profile['longitude'].iloc[0],
                              ypred - this_profile['latitude'].iloc[0],
                              length_includes_head=True, fc='k', ec='c', head_width=0.025, zorder=10)
                this_ax.plot(xpred, ypred, 'k+', zorder=10)

            this_ax.set_title("")
            # this_ax.set_ylabel("Cycle %i predictions" % (i_cycle+1))
            this_ax.set_title("%s\nCycle %i predictions" % (title, self.sim_cycles[i_cycle]), fontsize=6)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi,
                               subplot_kw={'projection': ccrs.PlateCarree()},
                               sharex=True, sharey=True)
        ax, ix = ax.flatten(), -1

        if orient == 'portrait':
            rows = range(self.n_cycles)
            cols = [1, 2]
        else:
            rows = [1, 2]
            cols = range(self.n_cycles)

        if orient == 'portrait':
            for i_cycle in rows:
                for ip in cols:
                    ix += 1
                    plot_this(ax[ix], i_cycle, ip)
        else:
            for ip in rows:
                for i_cycle in cols:
                    ix += 1
                    plot_this(ax[ix], i_cycle, ip)

        # log.debug("Start to write metrics string")
        #
        # xpred = SP.prediction[i_cycle + 1]['location']['longitude']['value']
        #
        # err = recovery['prediction_location_error']
        # met = recovery['prediction_metrics']
        # if this_profile.shape[0] > 1:
        #     # err_str = "Prediction vs Truth: [%0.2fkm, $%0.2f^o$]" % (err['distance'], err['bearing'])
        #     err_str = "Prediction errors: [dist=%0.2f%s, bearing=$%0.2f^o$, time=%s]\n" \
        #               "Distance error represents %s of transit at 12kt" % (err['distance']['value'],
        #                                               err['distance']['unit'],
        #                                               err['bearing']['value'],
        #                                               strfdelta(pd.Timedelta(err['time']['value'], 'h'),
        #                                                         "{hours}H{minutes:02d}"),
        #                                               strfdelta(pd.Timedelta(met['transit']['value'], 'h'),
        #                                                         "{hours}H{minutes:02d}"))
        # else:
        #     err_str = ""
        #
        # fig.suptitle("VirtualFleet recovery prediction for WMO %i: \
        # starting from cycle %i, predicting cycle %i\n%s\n%s\n%s" %
        #              (wmo, cyc[0], cyc[1], get_cfg_str(cfg), err_str, "Prediction based on %s" % vel_name), fontsize=15)

        plt.tight_layout()
        if save_figure:
            save_figurefile(fig, 'vfrecov_predictions_%s' % sim_suffix, workdir)

        return fig, ax


class SimPredictor(SimPredictor_3):

    def to_json(self, fp=None):
        kw = {'indent': 4, 'sort_keys': True, 'default': str}
        if fp is not None:
            if hasattr(fp, 'write'):
                json.dump(self._json, fp, **kw)
            else:
                with open(fp, 'w') as f:
                    json.dump(self._json, f, **kw)
        else:
            results_js = json.dumps(self._json, **kw)
            return results_js


def get_ea_profile_page_url(wmo, cyc):
    try:
        url = argoplot.dashboard(wmo, cyc, url_only=True)
    except:
        log.info("EA dashboard page not available for this profile: %i/%i" % (wmo, cyc))
        url = "404"
    return url


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


def get_sim_suffix(this_args, this_cfg):
    """Compose a string suffix for output files"""
    # suf = '%s_%i' % (this_args.velocity, this_args.nfloats)
    suf = 'VEL%s_NF%i_CYCDUR%i_PARKD%i_PROFD%i_SFD%i' % (this_args.velocity,
                                                this_args.nfloats,
                                                int(this_cfg.mission['cycle_duration']),
                                                int(this_cfg.mission['parking_depth']),
                                                int(this_cfg.mission['profile_depth']),
                                                int(this_cfg.mission['reco_free_surface_drift']))
    return suf


def predictor(args):
    """Prediction manager"""
    execution_start = time.time()
    process_start = time.process_time()

    if is_wmo(args.wmo):
        WMO = args.wmo
    if is_cyc(args.cyc):
        CYC = [check_cyc(args.cyc)[0]-1]
        [CYC.append(c) for c in check_cyc(args.cyc)]
    if args.velocity not in ['ARMOR3D', 'GLORYS']:
        raise ValueError("Velocity field must be one in: ['ARMOR3D', 'GLORYS']")
    else:
        VEL_NAME = args.velocity.upper()

    puts('CYC = %s' % CYC, color=COLORS.magenta)
    # raise ValueError('stophere')

    if args.save_figure:
        mplbackend = matplotlib.get_backend()
        matplotlib.use('Agg')

    # Where do we find the VirtualFleet repository ?
    if not args.vf:
        if os.uname()[1] == 'data-app-virtualfleet-recovery':
            euroargodev = os.path.expanduser('/home/ubuntu')
        else:
            euroargodev = os.path.expanduser('~/git/github/euroargodev')
    else:
        euroargodev = os.path.abspath(args.vf)
        if not os.path.exists(os.path.join(euroargodev, "VirtualFleet")):
            raise ValueError("VirtualFleet can't be found at '%s'" % euroargodev)

    # Import the VirtualFleet library
    sys.path.insert(0, os.path.join(euroargodev, "VirtualFleet"))
    from virtualargofleet import Velocity, VirtualFleet, FloatConfiguration, ConfigParam
    # from virtualargofleet.app_parcels import ArgoParticle

    # Set up the working directory:
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

    # Load these profiles' information:
    if not args.json:
        puts("\nYou can check this float dashboard while we prepare the prediction:")
        puts("\t%s" % argoplot.dashboard(WMO, url_only=True), color=COLORS.green)
        puts("\nLoading float profiles index ...")
    host = "https://data-argo.ifremer.fr"
    # host = "/home/ref-argo/gdac" if os.uname()[0] == 'Darwin' else "https://data-argo.ifremer.fr"
    # host = "/home/ref-argo/gdac" if not os.uname()[0] == 'Darwin' else "~/data/ARGO"
    THIS_PROFILE = store(host=host).search_wmo_cyc(WMO, CYC).to_dataframe()
    THIS_DATE = pd.to_datetime(THIS_PROFILE['date'].values[0], utc=True)
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

    if args.cfg_profile_depth is not None:
        puts("profile_depth=%i is overwritten with %i" % (CFG.mission['profile_depth'],
                                                          float(args.cfg_profile_depth)))
        CFG.update('profile_depth', float(args.cfg_profile_depth))

    CFG.params = ConfigParam(key='reco_free_surface_drift',
                             value=int(args.cfg_free_surface_drift),
                             unit='cycle',
                             description='First cycle with free surface drift',
                             dtype=int)

    # Save virtual float configuration on file:
    CFG.to_json(os.path.join(WORKDIR, "floats_configuration_%s.json" % get_sim_suffix(args, CFG)))

    if not args.json:
        puts("\n".join(["\t%s" % line for line in CFG.__repr__().split("\n")]), color=COLORS.green)

    # Get the cycling frequency (in days, this is more a period then...):
    CYCLING_FREQUENCY = int(np.round(CFG.mission['cycle_duration']/24))

    # Define domain to load velocity for, and get it:
    width = args.domain_size + np.abs(np.ceil(THIS_PROFILE['longitude'].values[-1] - CENTER[0]))
    height = args.domain_size + np.abs(np.ceil(THIS_PROFILE['latitude'].values[-1] - CENTER[1]))
    VBOX = [CENTER[0] - width / 2, CENTER[0] + width / 2, CENTER[1] - height / 2, CENTER[1] + height / 2]
    N_DAYS = (len(CYC)-1)*CYCLING_FREQUENCY+1
    if not args.json:
        puts("\nLoading %s velocity field to cover %i days..." % (VEL_NAME, N_DAYS))
    ds_vel, velocity_file = get_velocity_field(VBOX, THIS_DATE,
                                           n_days=N_DAYS,
                                           output=WORKDIR,
                                           dataset=VEL_NAME)
    VEL = Velocity(model='GLORYS12V1' if VEL_NAME == 'GLORYS' else VEL_NAME, src=ds_vel)
    if not args.json:
        puts("\n\t%s" % str(ds_vel), color=COLORS.green)
        puts("\n\tLoaded velocity field from %s to %s" %
             (pd.to_datetime(ds_vel['time'][0].values).strftime("%Y-%m-%dT%H:%M:%S"),
              pd.to_datetime(ds_vel['time'][-1].values).strftime("%Y-%m-%dT%H:%M:%S")), color=COLORS.green)
    figure_velocity(VBOX, VEL, VEL_NAME, THIS_PROFILE, WMO, CYC, save_figure=args.save_figure, workdir=WORKDIR)

    # raise ValueError('stophere')

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

    # Set up VirtualFleet:
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
    # if os.path.exists(output_path):
    #     shutil.rmtree(output_path)
    #
    # VFleet.simulate(duration=timedelta(hours=N_DAYS*24+1),
    #                 step=timedelta(minutes=5),
    #                 record=timedelta(minutes=30),
    #                 output=True,
    #                 output_folder=WORKDIR,
    #                 output_file='trajectories_%s.zarr' % get_sim_suffix(args, CFG),
    #                 verbose_progress=not args.json,
    #                 )

    # VirtualFleet, get simulated profiles index:
    if not args.json:
        puts("\nExtract swarm profiles index...")

    T = Trajectories(WORKDIR + "/" + 'trajectories_%s.zarr' % get_sim_suffix(args, CFG))
    DF_SIM = T.get_index().add_distances(origin=[THIS_PROFILE['longitude'].values[0], THIS_PROFILE['latitude'].values[0]])
    if not args.json:
        puts(str(T), color=COLORS.magenta)
        puts(DF_SIM.head().to_string(), color=COLORS.green)
    figure_positions(args, VEL, DF_SIM, DF_PLAN, THIS_PROFILE, CFG, WMO, CYC, VEL_NAME,
                     dd=1, save_figure=args.save_figure, workdir=WORKDIR)

    # Recovery, make predictions based on simulated profile density:
    SP = SimPredictor(DF_SIM, THIS_PROFILE)
    if not args.json:
        puts("\nPredict float cycle position(s) from swarm simulation...", color=COLORS.white)
        puts(str(SP), color=COLORS.magenta)
    SP.fit_predict()
    SP.add_metrics(VEL)
    SP.plot_predictions(VEL,
                         CFG,
                         sim_suffix=get_sim_suffix(args, CFG),
                         save_figure=args.save_figure,
                         workdir=WORKDIR,
                         orient='portrait')
    results = SP.predictions

    # Recovery, compute more swarm metrics:
    for this_cyc in T.sim_cycles:
        jsmetrics, fig, ax = T.analyse_pairwise_distances(cycle=this_cyc,
                                                          save_figure=True,
                                                          this_args=args,
                                                          this_cfg=CFG,
                                                          sim_suffix=get_sim_suffix(args, CFG),
                                                          workdir=WORKDIR,
                                                          )
        if 'metrics' in results['predictions'][this_cyc]:
            for key in jsmetrics.keys():
                results['predictions'][this_cyc]['metrics'].update({key: jsmetrics[key]})
        else:
            results['predictions'][this_cyc].update({'metrics': jsmetrics})

    # Recovery, finalize JSON output:
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

    with open(os.path.join(WORKDIR, 'prediction_%s.json' % get_sim_suffix(args, CFG)), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4, default=str, sort_keys=True)

    if not args.json:
        puts(results_js, color=COLORS.green)
        puts("\nCheck results at:")
        puts("\t%s" % WORKDIR, color=COLORS.green)

    if args.save_figure:
        plt.close('all')
        # Restore Matplotlib backend
        matplotlib.use(mplbackend)

    if not args.save_sim:
        shutil.rmtree(output_path)

    return results_js


if __name__ == '__main__':
    # Read mandatory arguments from the command line
    ARGS = setup_args().parse_args()

    js = predictor(ARGS)
    if ARGS.json:
        sys.stdout.write(js)

    # Exit gracefully
    sys.exit(0)
