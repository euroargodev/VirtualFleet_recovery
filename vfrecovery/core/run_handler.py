import pandas as pd
import numpy as np
from typing import List

from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from sklearn.metrics import pairwise_distances

import matplotlib
import matplotlib.pyplot as plt
import argopy.plot as argoplot
import cartopy.crs as ccrs

from vfrecovery.plots.utils import save_figurefile, map_add_features
from vfrecovery.utils.geo import haversine, bearing
from vfrecovery.json import Simulation, Profile, Location, Metrics, Transit, SurfaceDrift, Location_error


pp_obj = lambda x: "\n%s" % "\n".join(["\t%s" % line for line in x.__repr__().split("\n")])


class RunAnalyserCore:
    """

    Examples
    --------
    T = Trajectories(traj_zarr_file)
    df = T.get_index().add_distances()

    SP = RunAnalyser(df)
    SP.fit_predict()
    SP.add_metrics(VFvelocity)
    SP.bbox()
    SP.plot_predictions(VFvelocity)
    SP.plan
    SP.n_cycles
    SP.trajectory
    """

    def __init__(self, df_sim: pd.DataFrame, df_obs: pd.DataFrame):
        self.swarm = df_sim
        self.obs = df_obs
        # self.set_weights()
        self.WMO = np.unique(df_obs['wmo'])[0]
        self.jsobj = []

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
    def has_ref(self):
        return len(self.obs_cycles) > 1

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
        if len(self.jsobj.predictions) == 0:
            raise ValueError("Please call `fit_predict` first")

        traj_prediction = np.array([self.obs['longitude'].values[0],
                                    self.obs['latitude'].values[0],
                                    self.obs['date'].values[0]])[
            np.newaxis]  # Starting point where swarm was deployed
        for p in self.jsobj.predictions:
            xpred, ypred, tpred = p.location.longitude, p.location.latitude, p.location.time
            traj_prediction = np.concatenate((traj_prediction,
                                              np.array([xpred, ypred, tpred])[np.newaxis]),
                                             axis=0)
        return traj_prediction


    def bbox(self, s: float = 1) -> list:
        """Get a simulation bounding box

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


class RunAnalyserPredictor(RunAnalyserCore):

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

    def fit_predict(self, weights_scale: float = 20.) -> List[Profile]:
        """Predict profile positions from simulated float swarm

        Prediction is based on a :class:`klearn.neighbors._kde.KernelDensity` estimate of the N_FLOATS
        simulated, weighted by their deployment distance to the observed previous cycle position.

        Parameters
        ----------
        weights_scale: float (default=20)
            Scale (in deg) to use to weight the deployment distance to the observed previous cycle position

        Returns
        -------
        List[Profile]
        """

        # Compute weights of the swarm float profiles locations
        self.set_weights(scale=weights_scale)

        # self._prediction_data = {'weights_scale': weights_scale, 'cyc': {}}

        cycles = np.unique(self.swarm['cyc']).astype(int)  # 1, 2, ...
        Plist = []
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

            # Store results in a Profile instance:
            p = Profile.from_dict({
                'location': Location.from_dict({
                    'longitude': xpred,
                    'latitude': ypred,
                    'time': tpred,
                    'description': None,
                }),
                'wmo': int(self.WMO),
                'cycle_number': int(self.sim_cycles[icyc]),
                'virtual_cycle_number': int(this_sim_cyc),
                'description': "Simulated profile #%i" % this_sim_cyc,
                'metrics': Metrics.from_dict({'description': None}),
                'url_float': argoplot.dashboard(self.WMO, url_only=True),
            })
            Plist.append(p)

        # Store results internally
        obs_cyc = self.obs_cycles[0]
        this_df = self.obs[self.obs['cyc'] == obs_cyc]

        self.jsobj = Simulation.from_dict({
            "initial_profile": Profile.from_dict({
                'location': Location.from_dict({
                    'longitude': this_df['longitude'].iloc[0],
                    'latitude': this_df['latitude'].iloc[0],
                    'time': this_df['date'].iloc[0],
                    'description': None,
                }),
                'wmo': int(self.WMO),
                'cycle_number': int(obs_cyc),
                'description': "Initial profile (observed)",
                'url_float': argoplot.dashboard(self.WMO, url_only=True),
                'url_profile': argoplot.dashboard(self.WMO, obs_cyc, url_only=True),
            }),
            "predictions": Plist,
            "observations": None,
            "meta_data": None,
        })

        # Add more stuff to internal storage:
        self._add_ref()  # Fill: self.jsobj.observations
        self._predict_errors()  # Fill: self.jsobj.predictions.Metrics.error and self.jsobj.predictions.Metrics.transit
        #
        return self


class RunAnalyserDiagnostics(RunAnalyserPredictor):

    def _add_ref(self):
        """Possibly add observations data to internal data structure

        This is for past cycles, for which we have observed positions of the predicted profiles

        This populates the ``self.jsobj.observations`` property (``self.jsobj`` was created by the ``fit_predict`` method)

        """
        if len(self.jsobj.predictions) == 0:
            raise ValueError("Please call `fit_predict` first")

        # Observed profiles that were simulated:
        Plist = []
        for cyc in self.sim_cycles:
            if cyc in self.obs_cycles:
                this_df = self.obs[self.obs['cyc'] == cyc]
                p = Profile.from_dict({
                    'wmo': int(self.WMO),
                    'cycle_number': int(cyc),
                    'url_float': argoplot.dashboard(self.WMO, url_only=True),
                    'url_profile': argoplot.dashboard(self.WMO, cyc, url_only=True),
                    'location': Location.from_dict({'longitude': this_df['longitude'].iloc[0],
                                                    'latitude': this_df['latitude'].iloc[0],
                                                    'time': this_df['date'].iloc[0]})
                })
                Plist.append(p)

        self.jsobj.observations = Plist

        return self

    def _predict_errors(self):
        """Possibly compute error metrics for the predicted positions

        This is for past cycles, for which we have observed positions of the predicted profiles

        This populates the ``self.jsobj.predictions.Metrics.error`` and ``self.jsobj.predictions.Metrics.transit`` properties (``self.jsobj`` was created by the ``fit_predict`` method)

        A transit time to cover the distance error is also calculated
        (assume a 12 kts boat speed with 1 kt = 1.852 km/h)

        """
        if len(self.jsobj.predictions) == 0:
            raise ValueError("Please call `fit_predict` first")

        Plist_updated = []
        for p in self.jsobj.predictions:
            if p.cycle_number in self.obs_cycles:
                this_obs_profile = self.obs[self.obs['cyc'] == p.cycle_number]
                xobs = this_obs_profile['longitude'].iloc[0]
                yobs = this_obs_profile['latitude'].iloc[0]
                tobs = this_obs_profile['date'].iloc[0]

                prev_obs_profile = self.obs[self.obs['cyc'] == p.cycle_number - 1]
                xobs0 = prev_obs_profile['longitude'].iloc[0]
                yobs0 = prev_obs_profile['latitude'].iloc[0]

                xpred = p.location.longitude
                ypred = p.location.latitude
                tpred = p.location.time

                dd = haversine(xobs, yobs, xpred, ypred)

                observed_bearing = bearing(xobs0, yobs0, xobs, yobs)
                sim_bearing = bearing(xobs0, yobs0, xpred, ypred)

                dt = pd.Timedelta(tpred - tobs)# / np.timedelta64(1, 's')

                p.metrics.error = Location_error.from_dict({
                    'distance': np.round(dd, 3),
                    'bearing': np.round(sim_bearing - observed_bearing, 3),
                    'time': pd.Timedelta(dt, 'h')
                })

                # also compute a transit time to cover the distance error:
                p.metrics.transit = Transit.from_dict({
                    'value':
                        pd.Timedelta(p.metrics.error.distance / (12 * 1.852), 'h').seconds / 3600.
                })

            Plist_updated.append(p)

        self.jsobj.predictions = Plist_updated
        return self

    def add_metrics(self, VFvel=None):
        """Possibly compute more metrics to interpret the prediction error

        This is for past cycles, for which we have observed positions of the predicted profiles

        This populates the ``self.jsobj.predictions.Metrics.surface_drift`` property (``self.jsobj`` was created by the ``fit_predict`` method)

        1. Compute surface drift due to the time lag between the predicted profile timing and the expected one

        """
        # cyc0 = self.obs_cycles[0]
        if len(self.jsobj.predictions) == 0:
            raise ValueError("Please call `predict` first")

        Plist_updated = []
        for p in self.jsobj.predictions:
            if p.cycle_number in self.obs_cycles and isinstance(p.metrics.error, Location_error):
                # Compute the possible drift due to the time lag between the predicted profile timing and the expected one:
                if VFvel is not None:
                    xpred, ypred, tpred = p.location.longitude, p.location.latitude, p.location.time
                    dsc = VFvel.field.interp(
                        {VFvel.dim['lon']: xpred,
                         VFvel.dim['lat']: ypred,
                         VFvel.dim['time']: tpred,
                         VFvel.dim['depth']:
                             VFvel.field[{VFvel.dim['depth']: 0}][VFvel.dim['depth']].values[np.newaxis][0]}
                    )
                    velc = np.sqrt(dsc[VFvel.var['U']] ** 2 + dsc[VFvel.var['V']] ** 2).values[np.newaxis][0]  # m/s
                    p.metrics.surface_drift = SurfaceDrift.from_dict({
                        "surface_currents_speed": velc,  # m/s by default
                        "value": (np.abs(p.metrics.error.time.total_seconds()) * velc / 1e3)  # km
                    })

            Plist_updated.append(p)

        self.jsobj.predictions = Plist_updated
        return self


class RunAnalyserView(RunAnalyserDiagnostics):

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
                    figsize = (5, (self.n_cycles - 1) * 5)
        else:
            if self.n_cycles == 1:
                nrows, ncols = 1, 2
            else:
                nrows, ncols = 2, self.n_cycles
            if figsize is None:
                figsize = (ncols * 5, 5)

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

        plt.tight_layout()
        if save_figure:
            save_figurefile(fig, 'vfrecov_predictions_%s' % sim_suffix, workdir)

        return fig, ax


class RunAnalyser(RunAnalyserView):

    def to_json(self, fp=None):
        return self.jsobj.to_json(fp=fp)