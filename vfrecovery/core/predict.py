import time
from argopy.utils import is_wmo, is_cyc, check_cyc, check_wmo
import argopy.plot as argoplot
from virtualargofleet import Velocity, VirtualFleet, FloatConfiguration, ConfigParam
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import os
import logging
import json
import pprint
from datetime import timedelta

from vfrecovery.json import Profile, MetaData, MetaDataSystem, MetaDataComputation
from vfrecovery.utils.formatters import COLORS
from vfrecovery.downloaders import get_velocity_field
from .utils import df_obs2jsProfile, ArgoIndex2df_obs, ArgoIndex2jsProfile, get_simulation_suffix, get_domain
from .deployment_plan import setup_deployment_plan
from .trajfile_handler import Trajectories
from .run_handler import RunAnalyser

root_logger = logging.getLogger("vfrecovery_root_logger")
sim_logger = logging.getLogger("vfrecovery_simulation")

pp_obj = lambda x: "\n%s" % "\n".join(["\t%s" % line for line in x.__repr__().split("\n")])


class log_this:

    def __init__(self, txt, log_level):
        """Log text to simulation and possibly root logger(s)"""
        getattr(root_logger, log_level.lower())(txt)
        getattr(sim_logger, log_level.lower())(txt)

    @staticmethod
    def info(txt) -> 'log_this':
        return log_this(txt, 'INFO')

    @staticmethod
    def debug(txt) -> 'log_this':
        return log_this(txt, 'DEBUG')

    @staticmethod
    def warning(txt) -> 'log_this':
        return log_this(txt, 'WARNING')

    @staticmethod
    def error(txt) -> 'log_this':
        return log_this(txt, 'ERROR')


def setup_floats_config(
        wmo: int,
        cyc: int,
        cfg_parking_depth: float,
        cfg_cycle_duration: float,
        cfg_profile_depth: float,
        cfg_free_surface_drift: int,
) -> FloatConfiguration:
    """Load float configuration at a given cycle number and possibly overwrite data with user parameters"""
    log_this.debug("Loading float configuration...")
    try:
        CFG = FloatConfiguration([wmo, cyc])
    except:
        log_this.debug("Can't load this profile configuration, fall back on default values")
        CFG = FloatConfiguration('default')

    if cfg_parking_depth is not None:
        log_this.info("parking_depth=%i is overwritten with %i" % (CFG.mission['parking_depth'],
                                                                   float(cfg_parking_depth)))
        CFG.update('parking_depth', float(cfg_parking_depth))

    if cfg_cycle_duration is not None:
        log_this.info("cycle_duration=%i is overwritten with %i" % (CFG.mission['cycle_duration'],
                                                                    float(cfg_cycle_duration)))
        CFG.update('cycle_duration', float(cfg_cycle_duration))

    if cfg_profile_depth is not None:
        log_this.info("profile_depth=%i is overwritten with %i" % (CFG.mission['profile_depth'],
                                                                   float(cfg_profile_depth)))
        CFG.update('profile_depth', float(cfg_profile_depth))

    CFG.params = ConfigParam(key='reco_free_surface_drift',
                             value=int(cfg_free_surface_drift),
                             unit='cycle',
                             description='First cycle with free surface drift',
                             dtype=int)

    return CFG


class Simulation:
    """

    >>> S = Simulation(wmo, cyc, n_floats=n_floats, velocity=velocity)
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
        log_this.info("=" * 55)
        log_this.info("STARTING SIMULATION: WMO=%i / CYCLE_NUMBER=%i" % (wmo, cyc[1]))
        log_this.info("=" * 55)

        # log_this.info("n_predictions: %i" % n_predictions)
        log_this.info("Working with cycle numbers list: %s" % str(cyc))

        #
        url = argoplot.dashboard(wmo, url_only=True)
        txt = "You can check this float dashboard while we prepare the prediction: %s" % url
        log_this.info(txt)

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

        log_this.info("Loading float profiles index")
        self.P_obs, self.df_obs = ArgoIndex2jsProfile(self.wmo, self.cyc)
        [log_this.debug("Observed profiles list: %s" % pp_obj(p)) for p in self.P_obs]

        if len(self.P_obs) == 1:
            log_this.info('Real-time scenario: True position unknown !')
        else:
            log_this.info('Evaluation scenario: Historical position known')

    def _setup_float_config(self, **kwargs):
        """Load and setup float configuration"""

        # Load real float configuration at the previous cycle, to be used for the simulation as initial conditions.
        # (the loaded config is possibly overwritten with user defined cfg_* parameters)
        self.CFG = setup_floats_config(self.wmo, self.cyc[0],
                                       kwargs['cfg_parking_depth'],
                                       kwargs['cfg_cycle_duration'],
                                       kwargs['cfg_profile_depth'],
                                       kwargs['cfg_free_surface_drift'])
        self.MD.vfconfig = self.CFG  # Register floats configuration to the simulation meta-data class

        # and save the final virtual float configuration on file:
        self.CFG.to_json(
            Path(os.path.join(self.output_path, "floats_configuration_%s.json" % get_simulation_suffix(self.MD))))
        log_this.debug(pp_obj(self.CFG))

    def _setup_load_velocity_data(self, **kwargs):
        # Define domain to load velocity for:
        # In space:
        domain, domain_center = get_domain(self.P_obs, kwargs['domain_min_size'])
        # and time:
        cycle_period = int(np.round(self.CFG.mission['cycle_duration'] / 24))  # Get the float cycle period (in days)
        self.n_days = (len(self.cyc) - 1) * cycle_period + 1

        # log_this.info((domain_min_size, self.n_days))
        # log_this.info((domain_center, domain))
        log_this.info("Loading %s velocity field to cover %i days starting on %s" % (
            self.MD.velocity_field, self.n_days, self.P_obs[0].location.time))

        self.ds_vel, velocity_file = get_velocity_field(domain, self.P_obs[0].location.time,
                                                        n_days=self.n_days,
                                                        output=self.output_path,
                                                        dataset=self.MD.velocity_field)
        log_this.debug(pp_obj(self.ds_vel))
        log_this.info("Loaded %s field from %s to %s" % (
            self.MD.velocity_field,
            pd.to_datetime(self.ds_vel['time'][0].values).strftime("%Y-%m-%dT%H:%M:%S"),
            pd.to_datetime(self.ds_vel['time'][-1].values).strftime("%Y-%m-%dT%H:%M:%S"))
                      )

    def setup(self, **kwargs):
        """Fulfill all requirements for the simulation"""
        self._setup_load_observed_profiles()
        self._setup_float_config(**kwargs)
        self._setup_load_velocity_data(**kwargs)
        log_this.info("Simulation data will be registered with file suffix: '%s'" % get_simulation_suffix(self.MD))
        return self

    def _execute_get_plan(self):
        # VirtualFleet, get a deployment plan:
        log_this.info("Deployment plan setup")
        df_plan = setup_deployment_plan(self.P_obs[0], nfloats=self.MD.n_floats)
        log_this.info("Set %i virtual floats to deploy (i.e. swarm size = %i)" % (df_plan.shape[0], df_plan.shape[0]))

        self.PLAN = {'lon': df_plan['longitude'],
                     'lat': df_plan['latitude'],
                     'time': np.array([np.datetime64(t) for t in df_plan['date'].dt.strftime('%Y-%m-%d %H:%M').array]),
                     }

    def _execute_get_velocity(self):
        self.VEL = Velocity(model='GLORYS12V1' if self.MD.velocity_field == 'GLORYS' else self.MD.velocity_field,
                            src=self.ds_vel)
        # figure_velocity(VBOX, VEL, VEL_NAME, THIS_PROFILE, WMO, CYC, save_figure=args.save_figure, workdir=WORKDIR)

    def execute(self):
        """Execute a VirtualFleet simulation"""

        self._execute_get_velocity()
        self._execute_get_plan()

        # Set up VirtualFleet:
        log_this.info("VirtualFleet instance setup")
        self.VFleet = VirtualFleet(plan=self.PLAN,
                                   fieldset=self.VEL,
                                   mission=self.CFG)

        # Execute the simulation:
        log_this.info("Starting simulation")

        # Remove traj file if exists:
        # output_path = os.path.join(WORKDIR, 'trajectories_%s.zarr' % get_sim_suffix(args, CFG))
        # if os.path.exists(output_path):
        #     shutil.rmtree(output_path)

        self.traj_file = os.path.join(self.output_path, 'trajectories_%s.zarr' % get_simulation_suffix(self.MD))
        if os.path.exists(self.traj_file):
            log_this.info("Using data from a previous similar run (no simulation executed)")
        else:
            self.VFleet.simulate(duration=timedelta(hours=self.n_days * 24 + 1),
                                 step=timedelta(minutes=5),
                                 record=timedelta(minutes=30),
                                 output=True,
                                 output_folder=self.output_path,
                                 output_file='trajectories_%s.zarr' % get_simulation_suffix(self.MD),
                                 verbose_progress=True,
                                 )
            log_this.info("Simulation ended with success")
        return self

    def _predict_read_trajectories(self):

        # Get simulated profiles index:
        log_this.info("Extracting swarm profiles index")

        # self.traj = Trajectories(self.VFleet.output)
        self.traj = Trajectories(self.traj_file)
        self.traj.get_index().add_distances(origin=self.P_obs[0])
        log_this.debug(pp_obj(self.traj))

        # jsdata, fig, ax = self.traj.analyse_pairwise_distances(cycle=1, show_plot=True)

        # figure_positions(args, VEL, DF_SIM, DF_PLAN, THIS_PROFILE, CFG, WMO, CYC, VEL_NAME,
        #                  dd=1, save_figure=args.save_figure, workdir=WORKDIR)

    def _predict_positions(self):
        """Make predictions based on simulated profile density"""
        self.run = RunAnalyser(self.traj.to_index(), self.df_obs)
        log_this.info("Predicting float cycle position(s) from swarm simulation")
        log_this.debug(pp_obj(self.run))

        self.run.fit_predict()
        # SP.plot_predictions(VEL,
        #                      CFG,
        #                      sim_suffix=get_sim_suffix(args, CFG),
        #                      save_figure=args.save_figure,
        #                      workdir=WORKDIR,
        #                      orient='portrait')
        # results = self.run.predictions

    def predict(self):
        """Make float profile predictions based on the swarm simulation"""
        self._predict_read_trajectories()
        self._predict_positions()
        return self

    def _postprocess_metrics(self):
        log_this.info("Computing prediction metrics for past cycles with observed ground truth (possibly)")
        self.run.add_metrics(self.VEL)

    def _postprocess_swarm_metrics(self):
        log_this.info("Computing swarm metrics")
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
        })

        self.run_file = os.path.join(self.output_path, 'prediction_%s.json' % get_simulation_suffix(self.MD))
        self.to_json(fp=self.run_file)
        log_this.info("Simulation results and analysis saved in: %s" % self.run_file)

        log_this.info("VirtualFleet-Recovery prediction finished")
        return self

    def to_json(self, fp=None):
        y = self.run.jsobj  # :class:`Simulation` instance
        y.meta_data = self.MD
        return y.to_json(fp=fp)


def predict_function(
        wmo: int,
        cyc: int,
        velocity: str,
        n_predictions: int,
        output_path: Union[str, Path],
        cfg_parking_depth: float,
        cfg_cycle_duration: float,
        cfg_profile_depth: float,
        cfg_free_surface_drift: int,
        n_floats: int,
        domain_min_size: float,
        log_level: str,
) -> str:
    """
    Execute VirtualFleet-Recovery predictor and save results as a JSON string

    Parameters
    ----------
    wmo
    cyc
    velocity
    n_predictions
    output_path
    cfg_parking_depth
    cfg_cycle_duration
    cfg_profile_depth
    cfg_free_surface_drift
    n_floats
    domain_min_size
    log_level

    Returns
    -------
    str: a JSON formatted str

    """  # noqa
    if log_level == "QUIET":
        root_logger.disabled = True
        log_level = "CRITICAL"
    root_logger.setLevel(level=getattr(logging, log_level.upper()))
    # print('DEBUG', logging.DEBUG)
    # print('INFO', logging.INFO)
    # print('WARNING', logging.WARNING)
    # print('ERROR', logging.ERROR)
    # print('root_logger', root_logger.getEffectiveLevel())
    # print(root_logger.isEnabledFor(logging.INFO))

    execution_start = time.time()
    process_start = time.process_time()
    # run_id = pd.to_datetime('now', utc=True).strftime('%Y%m%d%H%M%S')

    # Validate arguments:
    assert is_wmo(wmo)
    assert is_cyc(cyc)
    wmo = check_wmo(wmo)[0]
    cyc = check_cyc(cyc)[0]

    if velocity.upper() not in ['ARMOR3D', 'GLORYS']:
        raise ValueError("Velocity field must be one in: ['ARMOR3D', 'GLORYS']")
    else:
        velocity = velocity.upper()

    # Build the list of cycle numbers to work with `cyc`:
    # The `cyc` list follows this structure:
    #   [PREVIOUS_CYCLE_USED_AS_INITIAL_CONDITIONS, CYCLE_NUMBER_REQUESTED_BY_USER, ADDITIONAL_CYCLE_i, ADDITIONAL_CYCLE_i+1, ...]
    # Prepend previous cycle number that will be used as initial conditions for the prediction of `cyc`:
    cyc = [cyc - 1, cyc]
    # Append additional `n_predictions` cycle numbers:
    [cyc.append(cyc[1] + n + 1) for n in range(n_predictions)]

    if output_path is None:
        # output_path = "vfrecovery_sims" % pd.to_datetime('now', utc=True).strftime("%Y%m%d%H%M%S")
        output_path = os.path.sep.join(["vfrecovery_simulations_data", str(wmo), str(cyc[1])])
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set-up simulation logger
    simlogfile = logging.FileHandler(os.path.join(output_path, "vfrecovery_simulations.log"), mode='a')
    simlogfile.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s:%(filename)s | %(message)s",
                                              datefmt='%Y/%m/%d %I:%M:%S'))
    sim_logger.handlers = []
    sim_logger.addHandler(simlogfile)
    # log_this.info("This is INFO")
    # log_this.warning("This is WARN")
    # log_this.debug("This is DEBUG")
    # log_this.error("This is ERROR")

    #
    S = Simulation(wmo, cyc,
                  n_floats=n_floats,
                  velocity=velocity,
                  output_path=output_path,
                  )
    S.setup(cfg_parking_depth=cfg_parking_depth,
            cfg_cycle_duration=cfg_cycle_duration,
            cfg_profile_depth=cfg_profile_depth,
            cfg_free_surface_drift=cfg_free_surface_drift,
            domain_min_size=domain_min_size,
            )
    S.execute()
    S.predict()
    S.postprocess()
    S.finish(execution_start, process_start)

    #
    # return S.MD.computation.to_json()
    # return MD.to_json()
    return S.to_json()

    # output = {'wmo': wmo, 'cyc': cyc, 'velocity': velocity, 'n_predictions': n_predictions, 'cfg': CFG.to_json(indent=0)}
    # json_dump = json.dumps(
    #     output, sort_keys=False, indent=2
    # )
    # return json_dump

# def predictor(args):
#     """Prediction manager"""
#
#     if args.save_figure:
#         mplbackend = matplotlib.get_backend()
#         matplotlib.use('Agg')

#     with open(os.path.join(WORKDIR, 'prediction_%s.json' % get_sim_suffix(args, CFG)), 'w', encoding='utf-8') as f:
#         json.dump(results, f, ensure_ascii=False, indent=4, default=str, sort_keys=True)
#
#     if not args.json:
#         puts(results_js, color=COLORS.green)
#         puts("\nCheck results at:")
#         puts("\t%s" % WORKDIR, color=COLORS.green)
#
#     if args.save_figure:
#         plt.close('all')
#         # Restore Matplotlib backend
#         matplotlib.use(mplbackend)
#
#     if not args.save_sim:
#         shutil.rmtree(output_path)
#
#     return results_js
#
