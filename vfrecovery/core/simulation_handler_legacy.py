import xarray as xr
import pandas as pd
import numpy as np
import json
import matplotlib
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import argopy.plot as argoplot
import cartopy.crs as ccrs

from vfrecovery.utils.misc import get_cfg_str, get_ea_profile_page_url
from vfrecovery.plots.utils import save_figurefile, map_add_features
from vfrecovery.utils.geo import haversine, bearing


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


