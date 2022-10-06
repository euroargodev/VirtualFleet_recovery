#!/usr/bin/env python
# -*coding: UTF-8 -*-
#
# This script can be used to make prediction of a specific float cycle position, given:
# - the previous cycle
# - the CMEMS GLORYS12 forecast at the time of the previous cycle
#
# This script is for testing the prediction system, and must be run on past float cycles.
#
# Created by gmaze on 06/10/2022
__author__ = 'gmaze@ifremer.fr'

import sys, os, glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import requests
from datetime import timedelta
from tqdm import tqdm
import argparse
import argopy
from argopy.stores.argo_index_pd import indexstore_pandas as store
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib
matplotlib.use('Agg')
from parcels import ParticleSet, FieldSet, Field
from abc import ABC


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

def puts(text, color=None, bold=False, file=sys.stdout):
    """Alternative to print, uses no color by default but accepts any color from the COLORS class."""
    if color is None:
        print(f'{PREF}{1 if bold else 0}m' + text + RESET, file=file)
    else:
        print(f'{PREF}{1 if bold else 0};{color}' + text + RESET, file=file)


def get_glorys_forecast_with_opendap(a_box, a_start_date, n_days=1):
    """Load a regional CMEMS forecast"""
    MOTU_USERNAME, MOTU_PASSWORD = (
        os.getenv("MOTU_USERNAME"),
        os.getenv("MOTU_PASSWORD"),
    )
    if not MOTU_USERNAME:
        raise ValueError("No MOTU_USERNAME in environment ! ")

    session = requests.Session()
    session.auth = (MOTU_USERNAME, MOTU_PASSWORD)
    # serverset = 'https://my.cmems-du.eu/thredds/dodsC/cmems_mod_glo_phy_my_0.083_P1D-m' # Daily
    # serverset = 'https://my.cmems-du.eu/thredds/dodsC/global-analysis-forecast-phy-001-024'
    # serverset = 'https://nrt.cmems-du.eu/thredds/dodsC/global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh' # Only surface fields
    serverset = 'https://nrt.cmems-du.eu/thredds/dodsC/global-analysis-forecast-phy-001-024-3dinst-uovo'  # 3D (uo, vo), 6-hourly
    # serverset = 'https://nrt.cmems-du.eu/thredds/dodsC/cmems_mod_glo_phy_anfc_merged-uv_PT1H-i' # hourly, surface

    store = xr.backends.PydapDataStore.open(serverset, session=session)
    ds = xr.open_dataset(store)
    # puts(ds.__repr__())

    # Get the starting date:
    t = "%0.4d-%0.2d-%0.2d %0.2d:00:00" % (a_start_date.year, a_start_date.month, a_start_date.day,
                                           np.array([0, 6, 12, 18])[
                                               np.argwhere(np.array([0, 6, 12, 18]) + 6 > a_start_date.hour)[0][0]])
    t = np.datetime64(pd.to_datetime(t))
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
    return glorys.load()


def get_velocity_field(a_box, a_date, n_days=1, root='.'):
    """Download or load the velocity field as netcdf/xarray"""
    velocity_file = os.path.join(root, 'velocity.nc')
    if not os.path.exists(velocity_file):
        # Load
        ds = get_glorys_forecast_with_opendap(a_box, a_date, n_days=n_days)
        # Save
        ds.to_netcdf(velocity_file)
    else:
        ds = xr.open_dataset(velocity_file)

    puts("\tLoaded velocity field from %s to %s" %
         (pd.to_datetime(ds['time'][0].values).strftime("%Y-%m-%dT%H:%M:%S"),
          pd.to_datetime(ds['time'][-1].values).strftime("%Y-%m-%dT%H:%M:%S")), color=COLORS.green)
    return ds, velocity_file


def get_HBOX(dd=1):
    # dd: how much to extend maps outward the deployment 'box'
    rx = DF_SIM['deploy_lon'].max() - DF_SIM['deploy_lon'].min()
    ry = DF_SIM['deploy_lat'].max() - DF_SIM['deploy_lat'].min()
    lonc, latc = DF_SIM['deploy_lon'].mean(), DF_SIM['deploy_lat'].mean()
    box = [lonc - rx / 2, lonc + rx / 2, latc - ry / 2, latc + ry / 2]
    ebox = [box[i] + [-dd, dd, -dd, dd][i] for i in range(0, 4)]  # Extended 'box'
    return ebox


def save_figure(this_fig, a_name):
    figname = os.path.join(WORKDIR, "%s.png" % a_name)
    this_fig.savefig(figname)
    return figname


def map_add_profiles(this_ax):
    this_ax.plot(THIS_PROFILE['longitude'][0], THIS_PROFILE['latitude'][0], 'k.', markersize=10, markeredgecolor='w')
    this_ax.plot(THIS_PROFILE['longitude'][1], THIS_PROFILE['latitude'][1], 'r.', markersize=10, markeredgecolor='w')
    this_ax.arrow(THIS_PROFILE['longitude'][0],
             THIS_PROFILE['latitude'][0],
             THIS_PROFILE['longitude'][1] - THIS_PROFILE['longitude'][0],
             THIS_PROFILE['latitude'][1] - THIS_PROFILE['latitude'][0],
             length_includes_head=True, fc='k', ec='k', head_width=0.05, zorder=10)

    return this_ax


def map_add_features(this_ax):
    argopy.plot.utils.latlongrid(this_ax)
    this_ax.add_feature(argopy.plot.utils.land_feature, edgecolor="black")
    return this_ax


def figure_velocity(ds_vel, box):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=90, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(box)
    ax = map_add_features(ax)
    ax = map_add_profiles(ax)

    ds_vel.isel(time=0, depth=0).plot.quiver(x="longitude", y="latitude", u="uo", v="vo", ax=ax, color='grey', alpha=0.5,
                                          add_guide=False)

    ax.set_title(
        "VirtualFleet recovery system for WMO %i: starting from cycle %i, predicting cycle %i\n1st level Velocity field" % (
        WMO, CYC[0], CYC[1]), fontsize=15);
    save_figure(fig, 'vfrecov_velocity')
    return None


def figure_positions(ds_vel, dd=1):
    ebox = get_HBOX(dd=dd)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 7), dpi=90,
                           subplot_kw={'projection': ccrs.PlateCarree()},
                           sharex=True, sharey=True)
    ax = ax.flatten()

    for ix in [0, 1, 2]:
        ax[ix].set_extent(ebox)
        ax[ix] = map_add_features(ax[ix])

        v = ds_vel.isel(time=0).interp(depth=CFG.mission['parking_depth']).plot.quiver(x="longitude", y="latitude", u="uo", v="vo", scale=20, ax=ax[ix], color='grey', alpha=0.5, add_guide=False)
        ax[ix].plot(DF_SIM['deploy_lon'], DF_SIM['deploy_lat'], '.', markersize=3, color='grey', alpha=0.1, markeredgecolor=None, zorder=0)
        if ix == 0:
            title = 'Velocity field at %0.2fm and deployment plan' % CFG.mission['parking_depth']
            v.set_alpha(1)
            # v.set_color('black')
        elif ix == 1:
            x, y = DF_SIM['longitude'], DF_SIM['latitude']
            title = 'Final float positions'
            sc = ax[ix].plot(x, y, '.', markersize=3, color='cyan', alpha=0.9, markeredgecolor=None)
        elif ix == 2:
            x, y = DF_SIM['rel_lon'], DF_SIM['rel_lat']
            title = 'Final floats position relative to last float position'
            sc = ax[ix].plot(x, y, '.', markersize=3, color='cyan', alpha=0.9, markeredgecolor=None)


        ax[ix] = map_add_profiles(ax[ix])
        ax[ix].set_title(title)

    fig.suptitle("VirtualFleet recovery prediction for WMO %i: starting from cycle %i, predicting cycle %i" % (WMO, CYC[0], CYC[1]), fontsize=15);
    save_figure(fig, "vfrecov_positions")
    return None


def figure_predictions(ds_vel, weights, bin_X, bin_Y, bin_res, Hrel, dd=1):
    ebox = get_HBOX(dd=dd)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,7), dpi=90,
                           subplot_kw={'projection': ccrs.PlateCarree()},
                           sharex=True, sharey=True)
    ax = ax.flatten()

    for ix in [0, 1, 2]:
        ax[ix].set_extent(ebox)
        ax[ix] = map_add_features(ax[ix])

        ds_vel.isel(time=0).interp(depth=CFG.mission['parking_depth']).plot.quiver(x="longitude", y="latitude", u="uo", v="vo", ax=ax[ix], color='grey', alpha=0.5, add_guide=False)

        ax[ix].plot(DF_SIM['deploy_lon'], DF_SIM['deploy_lat'], '.', markersize=3, color='grey', alpha=0.1, markeredgecolor=None, zorder=0)
        w = weights/np.max(np.abs(weights),axis=0)
        # cmap = plt.cm.cool
        cmap = plt.cm.Reds
        if ix == 0:
            x, y = DF_SIM['deploy_lon'], DF_SIM['deploy_lat']
            title = 'Inital float positions\ncolored with histogram weights'
            # wp = weights_plan/np.nanmax(np.abs(weights_plan),axis=0)
            sc = ax[ix].scatter(x, y, c=w, marker='o', s=4, alpha=w, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
        elif ix == 1:
            x, y = DF_SIM['longitude'], DF_SIM['latitude']
            title = 'Final float positions\ncolored with histogram weights'
            sc = ax[ix].scatter(x, y, c=w, marker='o', s=4, alpha=w, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
        elif ix == 2:
            x, y = DF_SIM['rel_lon'], DF_SIM['rel_lat']
            title = 'Final floats relative to last float position\ncolored with histogram weights'
            sc = ax[ix].scatter(x, y, c=w, marker='o', s=4, alpha=w, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
        elif ix == 3:
            # Hs = H/(np.nanmax(H)-np.nanmin(H))
            # Hs = Hrel/(np.nanmax(Hrel)-np.nanmin(Hrel))
            # sc = ax[ix].pcolor(bin_x[0:-1]+bin_res/2, bin_y[0:-1]+bin_res/2, Hrel.T, cmap=cmap, vmin=0, vmax=1)
            bin_X, bin_Y = np.meshgrid(bin_X[0:-1]+bin_res/2, bin_Y[0:-1]+bin_res/2)
            bin_X, bin_Y = bin_X.flatten(), bin_Y.flatten()
            c = (Hrel.T).flatten()
            alp = c/np.nanmax(np.abs(c),axis=0)
            alp[np.isnan(alp)] = 0
            sc = ax[ix].scatter(bin_X, bin_Y, c=c, marker='o', s=6, alpha=alp, edgecolor=None, vmin=0, vmax=1, cmap=cmap)
            title = 'Weighted profile density'

        plt.colorbar(sc, ax=ax[ix], shrink=0.5)

        ax[ix] = map_add_profiles(ax[ix])
        ax[ix].set_title(title)

    fig.suptitle("VirtualFleet recovery prediction for WMO %i: starting from cycle %i, predicting cycle %i" % (WMO, CYC[0], CYC[1]), fontsize=15);
    save_figure(fig, 'vfrecov_predictions')
    return None


class VelocityFieldProto(ABC):
    def plot(self):
        """Show ParticleSet"""
        temp_pset = ParticleSet(fieldset=self.fieldset,
                                pclass=ArgoParticle, lon=0, lat=0, depth=0)
        temp_pset.show(field=self.fieldset.U, with_particles=False)
        # temp_pset.show(field = self.fieldset.V,with_particles = False)


class VelocityField_Recovery_Forecast(VelocityFieldProto):
    """Velocity Field Helper for GLOBAL-ANALYSIS-FORECAST-PHY-001-024 product.

    Reference
    ---------
    https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSIS_FORECAST_PHY_001_024/DATA-ACCESS
    """

    def __init__(self, src, isglobal: bool = False, **kwargs):
        """

        Parameters
        ----------
        src: pattern
            Pattern to list netcdf source files
        isglobal : bool, default False
            Set to 1 if field is global, 0 otherwise
        """
        filenames = {'U': src, 'V': src}
        variables = {'U': 'uo', 'V': 'vo'}
        dimensions = {'time': 'time', 'depth': 'depth', 'lat': 'latitude', 'lon': 'longitude'}

        self.field = filenames  # Dictionary with 'U' and 'V' as keys and list of corresponding files as values
        self.var = variables  # Dictionary mapping 'U' and 'V' to netcdf VELocity variable names
        self.dim = dimensions  # Dictionary mapping 'time', 'depth', 'lat' and 'lon' to netcdf VELocity variable names
        self.isglobal = isglobal

        # define parcels fieldset
        self.fieldset = FieldSet.from_netcdf(
            self.field, self.var, self.dim,
            allow_time_extrapolation=True,
            time_periodic=False,
            deferred_load=True)

        if self.isglobal:
            self.fieldset.add_constant(
                'halo_west', self.fieldset.U.grid.lon[0])
            self.fieldset.add_constant(
                'halo_east', self.fieldset.U.grid.lon[-1])
            self.fieldset.add_constant(
                'halo_south', self.fieldset.U.grid.lat[0])
            self.fieldset.add_constant(
                'halo_north', self.fieldset.U.grid.lat[-1])
            self.fieldset.add_periodic_halo(zonal=True, meridional=True)

        # create mask for grounding management
        mask_file = glob.glob(self.field['U'])[0]
        ds = xr.open_dataset(mask_file)
        ds = eval("ds.isel("+self.dim['time']+"=0)")
        ds = ds[[self.var['U'],self.var['V']]].squeeze()

        mask = ~(ds.where((~ds[self.var['U']].isnull()) | (~ds[self.var['V']].isnull()))[
                 self.var['U']].isnull()).transpose(self.dim['lon'], self.dim['lat'], self.dim['depth'])
        mask = mask.values
        # create a new parcels field that's going to be interpolated during simulation
        self.fieldset.add_field(Field('mask', data=mask, lon=ds[self.dim['lon']].values, lat=ds[self.dim['lat']].values,
                                      depth=ds[self.dim['depth']].values,
                                      transpose=True, mesh='spherical', interp_method='nearest'))

    def __repr__(self):
        summary = ["<VelocityField.Recovery.GLOBAL_ANALYSIS_FORECAST_PHY_001_024>"]

        return "\n".join(summary)


def setup_deployment_plan(a_profile, a_date, nfloats=15000):
    # We will deploy a collection of virtual floats that are located around the real float with random perturbations in space and time

    # Amplitude of the profile position perturbations in the zonal (deg), meridional (deg), and temporal (hours) directions:
    rx = 2
    ry = 2
    rt = 6

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


def simu2index(df_plan, this_ds):
    # Specific method for the recovery simulations
    # This is very slow and could be optimized
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


def postprocess_index(this_df):
    # Compute some distances

    # Compute distance between the predicted profile and the initial profile location from the deployment plan
    # We assume that virtual floats are sequentially taken from the deployment plan
    # Since distances are very short, we compute a simple rectangular distance

    x2, y2 = THIS_PROFILE['longitude'].values[0], THIS_PROFILE['latitude'].values[0]  # real float initial position
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



if __name__ == '__main__':
    icons_help_string = """This script can be used to make prediction of a specific float cycle position.
    This script is for testing the prediction system, and must be run on past float cycles.
    Note that in order to download the velocity field from 'https://nrt.cmems-du.eu', you need to set the environment variables: MOTU_USERNAME and MOTU_PASSWORD.
    """

    parser = argparse.ArgumentParser(description='VirtualFleet recovery predictor',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="%s\n(c) Argo-France/Ifremer/LOPS, 2022" % icons_help_string)

    # Add long and short arguments
    parser.add_argument('wmo', help="Float WMO number", type=int)
    parser.add_argument("cyc", help="Cycle number to predict", type=int)
    parser.add_argument("--nfloats", help="Number of virtual floats used to make the prediction, default: 15000", type=int, default=15000)
    parser.add_argument("--output", help="Output folder, default: ./vfrecov/<WMO>/vfpred_<CYC>", default=None)
    parser.add_argument("--vf", help="Parent folder to the VirtualFleet repository clone", default=None)

    # Read mandatory arguments from the command line
    args = parser.parse_args()
    if argopy.utilities.is_wmo(args.wmo):
        WMO = args.wmo
    if argopy.utilities.is_cyc(args.cyc):
        CYC = [args.cyc-1, args.cyc]

    # Where do we find the VirtualFleet repository ?
    if not args.vf:
        euroargodev = os.path.expanduser('~/git/github/euroargodev')
    else:
        euroargodev = os.path.abspath(args.vf)
        if not os.path.exists(os.path.join(euroargodev, "VirtualFleet")):
            raise ValueError("VirtualFleet can't be found at '%s'" % euroargodev)

    # Import the VirtualFleet library
    sys.path.insert(0, os.path.join(euroargodev, "VirtualFleet"))
    from virtualargofleet import VirtualFleet, FloatConfiguration
    from virtualargofleet.app_parcels import ArgoParticle

    # Set-up the working directory:
    if not args.output:
        WORKDIR = os.path.sep.join([".", "vfrecov", str(WMO), "vfpred_%0.4d" % (CYC[1])])
    else:
        WORKDIR = os.path.sep.join([args.output, str(WMO), "vfpred_%0.4d" % (CYC[1])])
    WORKDIR = os.path.abspath(WORKDIR)
    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)

    # Load these profiles information:
    puts("\nYou can check the float dashboard here:")
    puts("\t%s" % argopy.plot.dashboard(WMO, url_only=True), color=COLORS.green)
    THIS_PROFILE = store().search_wmo_cyc(WMO, CYC).to_dataframe()
    THIS_DATE = pd.to_datetime(THIS_PROFILE['date'].values[0])
    puts("\nProfiles to work with:")
    puts(THIS_PROFILE.to_string())

    # Load real float configuration at the previous cycle:
    puts("\nLoading float configuration...")
    CFG = FloatConfiguration([WMO, CYC[0]])
    # CFG.update('cycle_duration', CYCLING_FREQUENCY * 24)
    puts(CFG.__repr__())

    # Get the cycling frequency (in days):
    # dt = pd.to_datetime(THIS_PROFILE['date'].values[1]) - pd.to_datetime(THIS_PROFILE['date'].values[0])
    # CYCLING_FREQUENCY = int(np.round(dt.days + dt.seconds / 86400))
    CYCLING_FREQUENCY = int(np.round(CFG.mission['cycle_duration'])/24)

    # Define domain to load velocity for, and get it:
    width = 10 + np.abs(np.ceil(THIS_PROFILE['longitude'].values[1] - THIS_PROFILE['longitude'].values[0]))
    height = 10 + np.abs(np.ceil(THIS_PROFILE['latitude'].values[1] - THIS_PROFILE['latitude'].values[0]))
    lonc, latc = THIS_PROFILE['longitude'].values[0], THIS_PROFILE['latitude'].values[0],
    VBOX = [lonc - width / 2, lonc + width / 2, latc - height / 2, latc + height / 2]
    puts("\nLoading velocity field for %i days..." % (CYCLING_FREQUENCY+2))
    ds, velocity_file = get_velocity_field(VBOX, THIS_DATE, n_days=CYCLING_FREQUENCY+2, root=WORKDIR)
    figure_velocity(ds, VBOX)

    # VirtualFleet, get a deployment plan:
    puts("\nVirtualFleet, get a deployment plan...")
    CENTER = [THIS_PROFILE['longitude'].values[0], THIS_PROFILE['latitude'].values[0]]
    DF_PLAN = setup_deployment_plan(CENTER, THIS_DATE, nfloats=args.nfloats)
    puts("\t%i virtual floats to deploy" % DF_PLAN.shape[0], color=COLORS.green)

    # VirtualFleet, set-up the fleet:
    puts("\nVirtualFleet, set-up the fleet...")
    VFleet = VirtualFleet(lat=DF_PLAN['latitude'],
                          lon=DF_PLAN['longitude'],
                          time=np.array([np.datetime64(t) for t in DF_PLAN['date'].dt.strftime('%Y-%m-%d %H:%M').array]),
                          vfield=VelocityField_Recovery_Forecast(velocity_file),
                          mission=CFG.mission)

    # VirtualFleet, execute the simulation:
    puts("\nVirtualFleet, execute the simulation...")
    VFleet.simulate(duration=timedelta(hours=CYCLING_FREQUENCY*24+12),
                    step=timedelta(minutes=5),
                    record=timedelta(seconds=3600/2),
                    output_folder=WORKDIR,
                    )

    # VirtualFleet, get simulated profiles index:
    puts("\nVirtualFleet, extract simulated profiles index...")
    ds_traj = xr.open_dataset(VFleet.run_params['output_file'])
    DF_SIM = simu2index(DF_PLAN, ds_traj)
    DF_SIM = postprocess_index(DF_SIM)
    puts(DF_SIM.head().to_string())
    figure_positions(ds, dd=1)

    # Recovery, make predictions based on simulated profile density:
    HBOX = get_HBOX(dd=1)
    bin_res = 1 / 12 / 2
    bin_x, bin_y = np.arange(HBOX[0], HBOX[1], bin_res), np.arange(HBOX[2], HBOX[3], bin_res)
    weights = np.exp(-(DF_SIM['distance_origin'] ** 2) / 0.15)
    weights[np.isnan(weights)] = 0
    H, xedges, yedges = np.histogram2d(DF_SIM['longitude'], DF_SIM['latitude'], bins=[bin_x, bin_y], weights=weights,
                                       density=True)
    Hrel, xedges, yedges = np.histogram2d(DF_SIM['rel_lon'], DF_SIM['rel_lat'], bins=[bin_x, bin_y], weights=weights,
                                          density=True)
    H[H == 0] = np.NaN
    Hrel[Hrel == 0] = np.NaN
    figure_predictions(ds, weights, bin_x, bin_y, bin_res, Hrel)

    puts("\nData saved in:")
    puts("\t%s" % WORKDIR, color=COLORS.green)