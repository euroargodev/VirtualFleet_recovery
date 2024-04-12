import xarray as xr
import pandas as pd
import numpy as np
import matplotlib
from scipy.signal import find_peaks
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from virtualargofleet import VelocityField
from pathlib import Path
import logging

from vfrecovery.utils.misc import get_cfg_str
from vfrecovery.plots.utils import map_add_features, save_figurefile
from vfrecovery.json import Profile, Location
from vfrecovery.json import Metrics, TrajectoryLengths, PairwiseDistances, PairwiseDistancesState


root_logger = logging.getLogger("vfrecovery_root_logger")


class default_logger:

    def __init__(self, txt, log_level):
        """Log text to simulation and possibly root logger(s)"""
        getattr(root_logger, log_level.lower())(txt)

    @staticmethod
    def info(txt) -> 'default_logger':
        return default_logger(txt, 'INFO')

    @staticmethod
    def debug(txt) -> 'default_logger':
        return default_logger(txt, 'DEBUG')

    @staticmethod
    def warning(txt) -> 'default_logger':
        return default_logger(txt, 'WARNING')

    @staticmethod
    def error(txt) -> 'default_logger':
        return default_logger(txt, 'ERROR')


class Trajectories:
    """Trajectory file manager for VFrecovery

    Examples:
    ---------
    T = Trajectories(traj_zarr_file)
    T.swarm_size
    T.sim_cycles
    df = T.to_index()
    df = T.get_index().add_distances()
    jsdata, fig, ax = T.analyse_pairwise_distances(cycle=1, save_figure=True)
    """

    def __init__(self, zfile, **kwargs):
        self.zarr_file = zfile
        self.obj = xr.open_zarr(zfile)
        self._index = None
        self.logger = default_logger if 'logger' not in kwargs else kwargs['logger']

    @property
    def swarm_size(self):
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
        summary.append("Swarm size: %i floats" % self.swarm_size)
        start_date = pd.to_datetime(self.obj['time'].isel(trajectory=0, obs=0).values)
        end_date = pd.to_datetime(self.obj['time'].isel(trajectory=0, obs=-1).values)
        summary.append("Simulation length: %s, from %s to %s" % (
            pd.Timedelta(end_date - start_date, 'd'), start_date.strftime("%Y/%m/%d"), end_date.strftime("%Y/%m/%d")))
        return "\n".join(summary)

    # def to_index_par(self) -> pd.DataFrame:
    #     # Deployment loc:
    #     deploy_lon, deploy_lat = self.obj.isel(obs=0)['lon'].values, self.obj.isel(obs=0)['lat'].values
    #
    #     def worker(ds, cyc, x0, y0):
    #         mask = np.logical_and((ds['cycle_number'] == cyc).compute(),
    #                               (ds['cycle_phase'] >= 3).compute())
    #         this_cyc = ds.where(mask, drop=True)
    #         if len(this_cyc['time']) > 0:
    #             data = {
    #                 'date': this_cyc.isel(obs=-1)['time'].values,
    #                 'latitude': this_cyc.isel(obs=-1)['lat'].values,
    #                 'longitude': this_cyc.isel(obs=-1)['lon'].values,
    #                 'wmo': 9000000 + this_cyc.isel(obs=-1)['trajectory'].values,
    #                 'cyc': cyc,
    #                 # 'cycle_phase': this_cyc.isel(obs=-1)['cycle_phase'].values,
    #                 'deploy_lon': x0,
    #                 'deploy_lat': y0,
    #             }
    #             return pd.DataFrame(data)
    #         else:
    #             return None
    #
    #     cycles = np.unique(self.obj['cycle_number'])
    #     rows = []
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         future_to_url = {
    #             executor.submit(
    #                 worker,
    #                 self.obj,
    #                 cyc,
    #                 deploy_lon,
    #                 deploy_lat
    #             ): cyc
    #             for cyc in cycles
    #         }
    #         futures = concurrent.futures.as_completed(future_to_url)
    #         for future in futures:
    #             data = None
    #             try:
    #                 data = future.result()
    #             except Exception:
    #                 raise
    #             finally:
    #                 rows.append(data)
    #
    #     rows = [r for r in rows if r is not None]
    #     df = pd.concat(rows).reset_index()
    #     df['wmo'] = df['wmo'].astype(int)
    #     df['cyc'] = df['cyc'].astype(int)
    #     # df['cycle_phase'] = df['cycle_phase'].astype(int)
    #     self._index = df
    #
    #     return self._index

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
                mask_end_of_cycle = np.logical_or((ds['cycle_phase'] == 3).compute(), (ds['cycle_phase'] == 4).compute())

                mask = np.logical_and((ds['cycle_number'] == cyc).compute(), mask_end_of_cycle)
                this_cyc = ds.where(mask, drop=True)

                if len(this_cyc['time']) > 0:

                    # Check if we didn't lose some particles:
                    n = len(x0) - len(this_cyc.isel(obs=-1)['time'].values)
                    if n > 0:
                        raise ValueError("%i virtual floats did not make all required cycles. They probably reached "
                                         "the edge of the velocity field domain. You should try to increase the domain "
                                         "size of the simulation." % n)
                    else:
                        data = {
                            'date': this_cyc.isel(obs=-1)['time'].values,
                            'latitude': this_cyc.isel(obs=-1)['lat'].values,
                            'longitude': this_cyc.isel(obs=-1)['lon'].values,
                            'wmo': 9000000 + this_cyc.isel(obs=-1)['trajectory'].values,
                            'cyc': this_cyc.isel(obs=-1)['cycle_number'].values,
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
                if ~ np.isnan(cyc):
                    df = worker(self.obj, cyc, deploy_lon, deploy_lat)
                    rows.append(df)
            rows = [r for r in rows if r is not None]
            if len(rows) > 0:
                df = pd.concat(rows).reset_index()
                df['wmo'] = df['wmo'].astype(int)
                df['cyc'] = df['cyc'].astype(int)
                # df['cycle_phase'] = df['cycle_phase'].astype(int)
                self._index = df
            else:
                raise ValueError("")

        return self._index

    def get_index(self):
        """Compute index and return self"""
        self.to_index()
        return self

    @property
    def index(self):
        self.get_index()
        return self._index

    def add_distances(self, origin: Location = None) -> pd.DataFrame:
        """Compute profiles distance to some origin

        Parameters
        ----------
        origin: :class:`Location`

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

        df = self.index

        x2, y2 = origin.longitude, origin.latitude  # real float initial position
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

        return self.index

    def analyse_pairwise_distances(self,
                                   virtual_cycle_number: int = 1,
                                   save_figure: bool = False,
                                   workdir: str = '.',
                                   sim_suffix=None,
                                   mplbackend: str = 'Agg',
                                   this_cfg=None,
                                   this_args: dict = None,
                                   ):

        def pairs_pdf(longitude, latitude):
            Xi = np.array((longitude, latitude)).T
            di = pairwise_distances(Xi, n_jobs=-1)
            di = np.triu(di)
            di[di == 0] = np.nan

            xi = di.flatten()
            xi = xi[~np.isnan(xi)]
            xi = xi[:, np.newaxis]

            histi, bin_edgesi = np.histogram(xi, bins=100, density=1)
            dhi = np.diff(bin_edgesi[0:2])
            peaksi, _ = find_peaks(histi / np.max(histi), height=.4, distance=20)

            return histi, bin_edgesi, peaksi, dhi, di

        # def get_hist_and_peaks(this_d):
        #     x = this_d.flatten()
        #     x = x[~np.isnan(x)]
        #     x = x[:, np.newaxis]
        #     hist, bin_edges = np.histogram(x, bins=100, density=1)
        #     # dh = np.diff(bin_edges[0:2])
        #     peaks, _ = find_peaks(hist / np.max(hist), height=.4, distance=20)
        #     return {'pdf': hist, 'bins': bin_edges[0:-1], 'Npeaks': len(peaks)}

        # Squeeze traj file to virtual_cycle_number (sim can have more than 1 cycle):
        ds = self.obj.where((self.obj['cycle_number'] == virtual_cycle_number).compute(), drop=True)
        ds = ds.compute()

        # Compute swarm trajectories relative to the single/only real float initial position:
        # (Make all swarm trajectories to start at the same first position)
        lon0, lat0 = self.obj.isel(obs=0)['lon'].values[0], self.obj.isel(obs=0)['lat'].values[
            0]  # deployment locations
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
        hist0, bin_edges0, peaks0, dh0, d0 = pairs_pdf(X['lon'].values, X['lat'].values)

        # Compute final points pairwise distances, PDF and nb of peaks:
        X = ds.isel(obs=-1)
        X = X.isel(trajectory=~np.isnan(X['lon']))
        dsf = X
        hist, bin_edges, peaks, dh, d = pairs_pdf(X['lon'].values, X['lat'].values)

        # Compute final points pairwise distances (relative traj), PDF and nb of peaks:
        X1 = ds.isel(obs=-1)
        X1 = X1.isel(trajectory=~np.isnan(X1['lonc']))
        dsfc = X1
        hist1, bin_edges1, peaks1, dh1, d1 = pairs_pdf(X1['lonc'].values, X1['latc'].values)

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

        if np.isinf(overlapping / len(peaks)):
            raise ValueError("Can't compute the prediction score, infinity !")

        # Store metrics as VFRschema instance
        PD = PairwiseDistances.from_dict({
            'description': None,
            'initial_state': PairwiseDistancesState.from_dict({
                'median': np.nanmedian(d0), 'std': np.nanstd(d0), 'nPDFpeaks': len(peaks0), 'description': None,
            }),
            'final_state': PairwiseDistancesState.from_dict({
                'median': np.nanmedian(d), 'std': np.nanstd(d), 'nPDFpeaks': len(peaks), 'description': None,
            }),
            'relative_state': PairwiseDistancesState.from_dict({
                'median': np.nanmedian(d1), 'std': np.nanstd(d1), 'nPDFpeaks': len(peaks1), 'description': None,
            }),
            'overlapping': overlapping,
            'staggering': staggering,
            'score': overlapping / len(peaks),
        })
        PD.std_ratio = PD.final_state.std / PD.initial_state.std

        M = Metrics.from_dict({
            "description": None,
            "trajectory_lengths": TrajectoryLengths.from_dict({
                "median": np.nanmedian(ds['length'].values),
                "std": np.nanstd(ds['length'].values),
                "description": None,
            }),
            "pairwise_distances": PD,
        })

        # Figure:
        if save_figure:
            initial_mplbackend = matplotlib.get_backend()
            matplotlib.use(mplbackend)

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
                line0 = "VirtualFleet recovery swarm simulation for cycle %i" % virtual_cycle_number
                line1 = "Simulation made with %i virtual floats" % (self.swarm_size)

            fig.suptitle("%s\n%s" % (line0, line1), fontsize=15)
            plt.tight_layout()

            if sim_suffix is not None:
                filename = 'vfrecov_metrics01_%s_cyc%i' % (sim_suffix, virtual_cycle_number)
            else:
                filename = 'vfrecov_metrics01_cyc%i' % (virtual_cycle_number)
            save_figurefile(fig, filename, workdir)

            # Rewind mpl backend to initial position:
            matplotlib.use(initial_mplbackend)

        # Exit
        if save_figure:
            return M, fig, ax
        else:
            return M

    def HBOX(self, s: float = 1.):
        """Swarm bounding box

        Parameters
        ----------
        s: float
            Set how much to extend maps outward the deployment 'box'

        Returns
        -------
        list
        """
        df_plan = self.index.iloc[0]

        box = [np.min([self.index['deploy_lon'].min(), self.index['longitude'].min(), self.index['rel_lon'].min()]),
               np.max([self.index['deploy_lon'].max(), self.index['longitude'].max(), self.index['rel_lon'].max()]),
               np.min([self.index['deploy_lat'].min(), self.index['latitude'].min(), self.index['rel_lat'].min()]),
               np.max([self.index['deploy_lat'].max(), self.index['latitude'].max(), self.index['rel_lat'].max()])]
        rx, ry = df_plan['longitude'].max() - df_plan['longitude'].min(), df_plan['latitude'].max() - df_plan[
            'latitude'].min()
        r = np.min([rx, ry])
        ebox = [box[0] - s * r, box[1] + s * r, box[2] - s * r, box[3] + s * r]
        return ebox

    def plot_positions(self,
                       domain_scale: float = 1,
                       vel: VelocityField = None,
                       vel_depth: float = 0.,
                       save: bool = True,
                       workdir: Path = Path('.'),
                       fname: str = 'swarm_positions',
                       mplbackend: str = 'Agg',
                       ):
        """

        >>> T = Trajectories(traj_file)
        >>> T.plot_positions(vel_depth=cfg.mission['parking_depth'])
        """
        import cartopy.crs as ccrs

        initial_mplbackend = matplotlib.get_backend()
        matplotlib.use(mplbackend)

        ebox = self.HBOX(s=domain_scale)

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 7), dpi=120,
                               subplot_kw={'projection': ccrs.PlateCarree()},
                               sharex=True, sharey=True)
        ax = ax.flatten()

        for ix in [0, 1, 2]:
            ax[ix].set_extent(ebox)
            ax[ix] = map_add_features(ax[ix])

            if vel is not None:
                vel.field.isel(time=0 if ix == 0 else -1).interp(depth=vel_depth).plot.quiver(x="longitude",
                                                                           y="latitude",
                                                                           u=vel.var['U'],
                                                                           v=vel.var['V'],
                                                                           ax=ax[ix],
                                                                           color='grey',
                                                                           alpha=1 if ix == 0 else 0.5,
                                                                           add_guide=False)

            ax[ix].plot(self.index['deploy_lon'], self.index['deploy_lat'], '.',
                        markersize=3, color='grey', alpha=0.1, markeredgecolor=None, zorder=0)
            if ix == 0:
                title = 'Initial Velocity field at %0.2fm and deployment plan' % vel_depth
            elif ix == 1:
                x, y, c = self.index['longitude'], self.index['latitude'], self.index['cyc']
                title = 'Final float positions'
                # sc = ax[ix].plot(x, y, '.', markersize=3, color='cyan', alpha=0.9, markeredgecolor=None)
                sc = ax[ix].scatter(x, y, c=c, s=3, alpha=0.9, edgecolors=None)
            elif ix == 2:
                x, y, c = self.index['rel_lon'], self.index['rel_lat'], self.index['cyc']
                title = 'Final floats position relative to last float position'
                # sc = ax[ix].plot(x, y, '.', markersize=3, color='cyan', alpha=0.9, markeredgecolor=None)
                sc = ax[ix].scatter(x, y, c=c, s=3, alpha=0.9, edgecolors=None)

            # ax[ix] = map_add_profiles(ax[ix], this_profile)
            ax[ix].set_title(title)

        # fig.suptitle("VirtualFleet recovery prediction for WMO %i: starting from cycle %i, predicting cycle %s\n%s" %
        #              (wmo, cyc[0], cyc[1:], get_cfg_str(cfg)), fontsize=15)
        plt.tight_layout()
        if save:
            save_figurefile(fig, fname, workdir)

        # Rewind mpl backend to initial position:
        matplotlib.use(initial_mplbackend)

        return fig, ax
