import time
import argopy.plot as argoplot
from virtualargofleet import Velocity, VirtualFleet
from pathlib import Path
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import logging

from vfrecovery.json import MetaData, MetaDataSystem, MetaDataComputation
from vfrecovery.downloaders import get_velocity_field
from .utils import ArgoIndex2jsProfile, get_simulation_suffix, get_domain, pp_obj
from .floats_config import setup_floats_config
from .deployment_plan import setup_deployment_plan
from .trajfile_handler import Trajectories
from .analysis_handler import RunAnalyser

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


class Simulation:
    """Base class to execute the simulation/prediction workflow

    >>> S = Simulation(wmo, cyc, n_floats=n_floats, velocity=velocity, output_path=Path('.'))
    >>> S.setup()
    >>> S.execute()
    >>> S.predict()
    >>> S.postprocess()
    >>> S.to_json()
    """

    def __init__(self, wmo, cyc, **kwargs):
        self.wmo = wmo
        self.cyc = cyc
        self.output_path = kwargs['output_path']
        self.logger = default_logger if 'logger' not in kwargs else kwargs['logger']

        self.overwrite = kwargs['overwrite'] if 'overwrite' in kwargs else False
        self.lazy = kwargs['lazy'] if 'lazy' in kwargs else True

        self.logger.info("%s \\" % ("=" * 55))
        self.logger.info("STARTING SIMULATION: WMO=%i / CYCLE_NUMBER=%i" % (self.wmo, self.cyc[1]))

        # self.logger.info("n_predictions: %i" % n_predictions)
        self.logger.info("Working with cycle numbers list: %s" % str(cyc))

        #
        url = argoplot.dashboard(wmo, url_only=True)
        txt = "You can check this float dashboard while we prepare the prediction: %s" % url
        self.logger.info(txt)

        # Create Simulation Meta-data class holder
        self.MD = MetaData.from_dict({
            'n_floats': kwargs['n_floats'],
            'velocity_field': kwargs['velocity'],
            'system': MetaDataSystem.auto_load(),
            'vfconfig': None,  # will be filled later
            'computation': None,  # will be filled later
        })

    def _setup_load_observed_profiles(self):
        """Load observed float profiles index"""

        self.logger.info("Loading float profiles index")
        self.P_obs, self.df_obs = ArgoIndex2jsProfile(self.wmo, self.cyc, cache=False, cachedir=str(self.output_path))
        [self.logger.debug("Observed profiles list: %s" % pp_obj(p)) for p in self.P_obs]

        if len(self.P_obs) == 1:
            self.logger.info('Real-time scenario: True position unknown !')
        else:
            self.logger.info('Evaluation scenario: Historical position known')

    def _setup_float_config(self, **kwargs):
        """Load and setup float configuration"""
        self.logger.info("Loading float configuration")

        # Load real float configuration at the previous cycle, to be used for the simulation as initial conditions.
        # (the loaded config is possibly overwritten with user defined cfg_* parameters)
        self.CFG = setup_floats_config(self.wmo, self.cyc[0],
                                       kwargs['cfg_parking_depth'],
                                       kwargs['cfg_cycle_duration'],
                                       kwargs['cfg_profile_depth'],
                                       kwargs['cfg_free_surface_drift'],
                                       self.logger,
                                       )
        self.MD.vfconfig = self.CFG  # Register floats configuration to the simulation meta-data class

        # and save the final virtual float configuration on file:
        self.CFG.to_json(
            Path(os.path.join(self.output_path, "floats_configuration_%s.json" % get_simulation_suffix(self.MD))))
        self.logger.debug(pp_obj(self.CFG))

    def _setup_load_velocity_data(self, **kwargs):
        # Define domain to load velocity for:
        # In space:
        domain, domain_center = get_domain(self.P_obs, kwargs['domain_min_size'])
        # and time:
        cycle_period = int(np.round(self.CFG.mission['cycle_duration'] / 24))  # Get the float cycle period (in days)
        self.n_days = len(self.cyc) * cycle_period + 1

        self.logger.info("Velocity field should cover %i cycles of %i hours" % (len(self.cyc), 24 * cycle_period))
        self.logger.info("Loading %i days of %s velocity starting on %s" % (
            self.n_days, self.MD.velocity_field, self.P_obs[0].location.time))

        self.ds_vel, velocity_file, new_file = get_velocity_field(domain, self.P_obs[0].location.time,
                                                                  n_days=self.n_days,
                                                                  output=self.output_path,
                                                                  dataset=self.MD.velocity_field,
                                                                  logger=self.logger,
                                                                  lazy=self.lazy,
                                                                  )
        if new_file:
            # We force overwrite results because we're using a new velocity field
            self.logger.warning("Found a new velocity field, force overwriting results")
            self.overwrite = True

        self.velocity_file = velocity_file
        self.logger.debug(pp_obj(self.ds_vel))
        self.logger.info("Loaded %s field from %s to %s" % (
            self.MD.velocity_field,
            pd.to_datetime(self.ds_vel['time'][0].values).strftime("%Y-%m-%dT%H:%M:%S"),
            pd.to_datetime(self.ds_vel['time'][-1].values).strftime("%Y-%m-%dT%H:%M:%S"))
                         )

    def setup(self, **kwargs):
        """Fulfill all requirements for the simulation"""
        self._setup_load_observed_profiles()
        self._setup_float_config(**kwargs)
        self._setup_load_velocity_data(**kwargs)
        self.logger.info("Simulation data will be registered under: %s%s*%s*" % (self.output_path,
                                                                                 os.path.sep,
                                                                                 get_simulation_suffix(self.MD)))
        self.logger.debug("Setup terminated")
        return self

    def _execute_get_velocity(self):
        self.logger.info("Create a velocity object")
        self.VEL = Velocity(model='GLORYS12V1' if self.MD.velocity_field == 'GLORYS' else self.MD.velocity_field,
                            src=self.ds_vel,
                            logger=self.logger,
                            )

        self.logger.info("Plot velocity")
        for it in [0, -1]:
            _, _, fname = self.VEL.plot(it=it, iz=0, save=True, workdir=self.output_path)
            fname.rename(
                str(fname).replace("velocity_%s" % self.VEL.name, Path(self.velocity_file).name.replace(".nc", ""))
            )

    def _execute_get_plan(self):
        # VirtualFleet, get a deployment plan:
        self.logger.info("Create a deployment plan")
        df_plan = setup_deployment_plan(self.P_obs[0], nfloats=self.MD.n_floats)
        self.logger.info(
            "Set %i virtual floats to deploy (i.e. swarm size = %i)" % (df_plan.shape[0], df_plan.shape[0]))

        self.PLAN = {'lon': df_plan['longitude'],
                     'lat': df_plan['latitude'],
                     'time': np.array([np.datetime64(t) for t in df_plan['date'].dt.strftime('%Y-%m-%d %H:%M').array]),
                     }

    def execute(self):
        """Execute a VirtualFleet simulation"""

        self._execute_get_velocity()
        self._execute_get_plan()

        # Set up VirtualFleet:
        self.logger.info("Create a VirtualFleet instance")
        self.VFleet = VirtualFleet(plan=self.PLAN,
                                   fieldset=self.VEL,
                                   mission=self.CFG,
                                   verbose_events=False)
        self.logger.debug(pp_obj(self.VFleet))

        # Execute the simulation:
        self.logger.info("Starting simulation")

        # Remove traj file if exists:
        # output_path = os.path.join(WORKDIR, 'trajectories_%s.zarr' % get_sim_suffix(args, CFG))
        # if os.path.exists(output_path):
        #     shutil.rmtree(output_path)

        self.traj_file = os.path.join(self.output_path, 'trajectories_%s.zarr' % get_simulation_suffix(self.MD))
        if os.path.exists(self.traj_file) and not self.overwrite:
            self.logger.warning("Using data from a previous similar run (no simulation executed)")
        else:
            self.VFleet.simulate(duration=timedelta(hours=self.n_days * 24 + 1),
                                 step=timedelta(minutes=5),
                                 record=timedelta(minutes=30),
                                 output=True,
                                 output_folder=self.output_path,
                                 output_file='trajectories_%s.zarr' % get_simulation_suffix(self.MD),
                                 verbose_progress=True,
                                 )
            self.logger.info("Simulation ended with success")
        self.logger.info(pp_obj(self.VFleet))
        return self

    def _predict_read_trajectories(self):

        # Get simulated profiles index:
        self.logger.info("Extract swarm profiles index")

        self.traj = Trajectories(self.traj_file)
        self.traj.get_index().add_distances(origin=self.P_obs[0])
        self.logger.debug(pp_obj(self.traj))

        self.logger.info("Plot swarm initial and final states")
        self.traj.plot_positions(domain_scale=2.,
                                 vel=self.VEL,
                                 vel_depth=self.CFG.mission['parking_depth'],
                                 save=True,
                                 workdir=self.output_path,
                                 fname='swarm_states_%s' % get_simulation_suffix(self.MD)
                                 )

    def _predict_positions(self):
        """Make predictions based on simulated profile density"""
        self.logger.info("Predict float cycle position(s) from swarm simulation")
        self.run = RunAnalyser(self.traj.index, self.df_obs)
        self.run.fit_predict()
        self.logger.debug(pp_obj(self.run))

        self.logger.info("Plot predictions")
        self.run.plot_predictions(
            vel=self.VEL,
            vel_depth=self.CFG.mission['parking_depth'],
            save=True,
            workdir=self.output_path,
            fname='predictions_%s' % get_simulation_suffix(self.MD),
            orient='portrait'
        )

    def predict(self):
        """Make float profile predictions based on the swarm simulation"""
        self._predict_read_trajectories()
        self._predict_positions()
        return self

    def _postprocess_metrics(self):
        if self.run.has_ref:
            self.logger.info("Computing prediction metrics for past cycles with observed ground truth")
        self.run.add_metrics(self.VEL)

    def _postprocess_swarm_metrics(self):
        self.logger.info("Computing swarm metrics")
        Plist_updated = []
        for p in self.run.jsobj.predictions:
            this_cyc = p.virtual_cycle_number
            swarm_metrics = self.traj.analyse_pairwise_distances(virtual_cycle_number=this_cyc, show_plot=False)
            p.metrics.trajectory_lengths = swarm_metrics.trajectory_lengths
            p.metrics.pairwise_distances = swarm_metrics.pairwise_distances
            Plist_updated.append(p)
        self.run.jsobj.predictions = Plist_updated

    def postprocess(self):
        self._postprocess_metrics()
        self._postprocess_swarm_metrics()
        return self

    def finish(self, execution_start: float, process_start: float):
        """Click timers and save results to finish"""
        self.MD.computation = MetaDataComputation.from_dict({
            'date': pd.to_datetime('now', utc=True),
            'wall_time': pd.Timedelta(time.time() - execution_start, 's'),
            'cpu_time': pd.Timedelta(time.process_time() - process_start, 's'),
            'description': None,
        })
        self.logger.debug(pp_obj(self.MD.computation))

        self.run_file = os.path.join(self.output_path, 'results_%s.json' % get_simulation_suffix(self.MD))
        self.to_json(fp=self.run_file)
        self.logger.info("Simulation results and analysis saved in: %s" % self.run_file)

        self.logger.info("END OF SIMULATION: WMO=%i / CYCLE_NUMBER=%i" % (self.wmo, self.cyc[1]))
        self.logger.info("%s /" % ("=" * 55))
        return self

    def to_json(self, fp=None):
        y = self.run.jsobj  # :class:`Simulation` instance
        y.meta_data = self.MD
        return y.to_json(fp=fp)
