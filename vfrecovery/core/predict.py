import time
from argopy.utils import is_wmo, is_cyc, check_cyc, check_wmo
import argopy.plot as argoplot
from virtualargofleet import Velocity, VirtualFleet, FloatConfiguration, ConfigParam
from pathlib import Path
from typing import Union
import pandas as pd
import os
import logging
import json

from vfrecovery.json import Profile, MetaData
from vfrecovery.utils.formatters import COLORS
from .utils import df_obs2jsProfile, ArgoIndex2df, ArgoIndex2JsProfile

root_logger = logging.getLogger("vfrecovery_root_logger")
sim_logger = logging.getLogger("vfrecovery_simulation")

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

    # Validate arguments:
    assert is_wmo(wmo)
    assert is_cyc(cyc)
    wmo = check_wmo(wmo)[0]
    cyc = check_cyc(cyc)[0]

    if velocity.upper() not in ['ARMOR3D', 'GLORYS']:
        raise ValueError("Velocity field must be one in: ['ARMOR3D', 'GLORYS']")
    else:
        velocity = velocity.upper()

    # Prepend previous cycle number that will be used as initial conditions for the prediction of `cyc`:
    cyc = [cyc - 1, cyc]

    if output_path is None:
        # output_path = "vfrecovery_sims" % pd.to_datetime('now', utc=True).strftime("%Y%m%d%H%M%S")
        output_path = os.path.sep.join(["vfrecovery_data", str(wmo), str(cyc[1])])
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set-up simulation logger
    simlogfile = logging.FileHandler(os.path.join(output_path, "vfpred.log"), mode='a')
    simlogfile.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s:%(filename)s:%(lineno)d | %(message)s",
                                              datefmt='%Y/%m/%d %I:%M:%S'))
    sim_logger.handlers = []
    sim_logger.addHandler(simlogfile)
    # log_this.info("This is INFO")
    # log_this.warning("This is WARN")
    # log_this.debug("This is DEBUG")
    # log_this.error("This is ERROR")
    log_this.info("\n\nSTARTING NEW SIMULATION: WMO=%i / CYCLE_NUMBER=%i\n" % (wmo, cyc[1]))

    #
    url = argoplot.dashboard(wmo, url_only=True)
    txt = "You can check this float dashboard while we prepare the prediction: %s" % url
    log_this.info(txt)

    # Load observed float profiles index
    log_this.debug("Loading float profiles index ...")
    df_obs = ArgoIndex2df(wmo, cyc)
    P_obs = df_obs2jsProfile(df_obs)
    # P_obs = ArgoIndex2JsProfile(wmo, cyc)
    # THIS_DATE = P_obs[0].location.time
    # CENTER = [P_obs[0].location.longitude, P_obs[0].location.latitude]

    log_this.debug("Profiles to work with:\n%s" % df_obs[['date', 'latitude', 'longitude', 'wmo', 'cyc', 'institution']].to_string(max_colwidth=35))
    if df_obs.shape[0] == 1:
        log_this.info('Real-case scenario: True position unknown !')
    else:
        log_this.info('Evaluation scenario: historical position known')

    # Load real float configuration at the previous cycle:
    log_this.debug("\nLoading float configuration...")
    try:
        CFG = FloatConfiguration([wmo, cyc[0]])
    except:
        log_this.info("Can't load this profile config, falling back on default values")
        CFG = FloatConfiguration('default')

    if cfg_parking_depth is not None:
        log_this.debug("parking_depth=%i is overwritten with %i" % (CFG.mission['parking_depth'],
                                                          float(cfg_parking_depth)))
        CFG.update('parking_depth', float(cfg_parking_depth))

    if cfg_cycle_duration is not None:
        log_this.debug("cycle_duration=%i is overwritten with %i" % (CFG.mission['cycle_duration'],
                                                          float(cfg_cycle_duration)))
        CFG.update('cycle_duration', float(cfg_cycle_duration))

    if cfg_profile_depth is not None:
        log_this.debug("profile_depth=%i is overwritten with %i" % (CFG.mission['profile_depth'],
                                                          float(cfg_profile_depth)))
        CFG.update('profile_depth', float(cfg_profile_depth))

    CFG.params = ConfigParam(key='reco_free_surface_drift',
                             value=int(cfg_free_surface_drift),
                             unit='cycle',
                             description='First cycle with free surface drift',
                             dtype=int)

    # Save virtual float configuration on file:
    # CFG.to_json(os.path.join(output_path, "floats_configuration_%s.json" % get_sim_suffix(args, CFG)))

    #     if not args.json:
    #         puts("\n".join(["\t%s" % line for line in CFG.__repr__().split("\n")]), color=COLORS.green)
    #


    #
    MD = MetaData.from_dict({
        'nfloats': 0,
        'velocity_field': velocity,
        'vfconfig': CFG,
        'computation': None
    })
    return MD.to_json()

    output = {'wmo': wmo, 'cyc': cyc, 'velocity': velocity, 'n_predictions': n_predictions, 'cfg': CFG.to_json(indent=0)}
    json_dump = json.dumps(
        output, sort_keys=False, indent=2
    )
    return json_dump




# def predictor(args):
#     """Prediction manager"""

#     if is_cyc(args.cyc):
#         CYC = [check_cyc(args.cyc)[0]-1]
#         [CYC.append(c) for c in check_cyc(args.cyc)]
#
#     puts('CYC = %s' % CYC, color=COLORS.magenta)
#     # raise ValueError('stophere')
#
#     if args.save_figure:
#         mplbackend = matplotlib.get_backend()
#         matplotlib.use('Agg')

#     # Load these profiles' information:
#     if not args.json:
#         puts("\nYou can check this float dashboard while we prepare the prediction:")
#         puts("\t%s" % argoplot.dashboard(WMO, url_only=True), color=COLORS.green)
#         puts("\nLoading float profiles index ...")
#     host = "https://data-argo.ifremer.fr"
#     # host = "/home/ref-argo/gdac" if os.uname()[0] == 'Darwin' else "https://data-argo.ifremer.fr"
#     # host = "/home/ref-argo/gdac" if not os.uname()[0] == 'Darwin' else "~/data/ARGO"
#     THIS_PROFILE = store(host=host).search_wmo_cyc(WMO, CYC).to_dataframe()
#     THIS_DATE = pd.to_datetime(THIS_PROFILE['date'].values[0], utc=True)
#     CENTER = [THIS_PROFILE['longitude'].values[0], THIS_PROFILE['latitude'].values[0]]
#     if not args.json:
#         puts("\nProfiles to work with:")
#         puts(THIS_PROFILE.to_string(max_colwidth=15), color=COLORS.green)
#         if THIS_PROFILE.shape[0] == 1:
#             puts('\nReal-case scenario: True position unknown !', color=COLORS.yellow)
#         else:
#             puts('\nEvaluation scenario: historical position known', color=COLORS.yellow)
#
#     # Load real float configuration at the previous cycle:
#     if not args.json:
#         puts("\nLoading float configuration...")
#     try:
#         CFG = FloatConfiguration([WMO, CYC[0]])
#     except:
#         if not args.json:
#             puts("Can't load this profile config, falling back on default values", color=COLORS.red)
#         CFG = FloatConfiguration('default')
#
#     if args.cfg_parking_depth is not None:
#         puts("parking_depth=%i is overwritten with %i" % (CFG.mission['parking_depth'],
#                                                           float(args.cfg_parking_depth)))
#         CFG.update('parking_depth', float(args.cfg_parking_depth))
#
#     if args.cfg_cycle_duration is not None:
#         puts("cycle_duration=%i is overwritten with %i" % (CFG.mission['cycle_duration'],
#                                                           float(args.cfg_cycle_duration)))
#         CFG.update('cycle_duration', float(args.cfg_cycle_duration))
#
#     if args.cfg_profile_depth is not None:
#         puts("profile_depth=%i is overwritten with %i" % (CFG.mission['profile_depth'],
#                                                           float(args.cfg_profile_depth)))
#         CFG.update('profile_depth', float(args.cfg_profile_depth))
#
#     CFG.params = ConfigParam(key='reco_free_surface_drift',
#                              value=int(args.cfg_free_surface_drift),
#                              unit='cycle',
#                              description='First cycle with free surface drift',
#                              dtype=int)
#
#     # Save virtual float configuration on file:
#     CFG.to_json(os.path.join(WORKDIR, "floats_configuration_%s.json" % get_sim_suffix(args, CFG)))
#
#     if not args.json:
#         puts("\n".join(["\t%s" % line for line in CFG.__repr__().split("\n")]), color=COLORS.green)
#
#     # Get the cycling frequency (in days, this is more a period then...):
#     CYCLING_FREQUENCY = int(np.round(CFG.mission['cycle_duration']/24))
#
#     # Define domain to load velocity for, and get it:
#     width = args.domain_size + np.abs(np.ceil(THIS_PROFILE['longitude'].values[-1] - CENTER[0]))
#     height = args.domain_size + np.abs(np.ceil(THIS_PROFILE['latitude'].values[-1] - CENTER[1]))
#     VBOX = [CENTER[0] - width / 2, CENTER[0] + width / 2, CENTER[1] - height / 2, CENTER[1] + height / 2]
#     N_DAYS = (len(CYC)-1)*CYCLING_FREQUENCY+1
#     if not args.json:
#         puts("\nLoading %s velocity field to cover %i days..." % (VEL_NAME, N_DAYS))
#     ds_vel, velocity_file = get_velocity_field(VBOX, THIS_DATE,
#                                            n_days=N_DAYS,
#                                            output=WORKDIR,
#                                            dataset=VEL_NAME)
#     VEL = Velocity(model='GLORYS12V1' if VEL_NAME == 'GLORYS' else VEL_NAME, src=ds_vel)
#     if not args.json:
#         puts("\n\t%s" % str(ds_vel), color=COLORS.green)
#         puts("\n\tLoaded velocity field from %s to %s" %
#              (pd.to_datetime(ds_vel['time'][0].values).strftime("%Y-%m-%dT%H:%M:%S"),
#               pd.to_datetime(ds_vel['time'][-1].values).strftime("%Y-%m-%dT%H:%M:%S")), color=COLORS.green)
#     figure_velocity(VBOX, VEL, VEL_NAME, THIS_PROFILE, WMO, CYC, save_figure=args.save_figure, workdir=WORKDIR)
#
#     # raise ValueError('stophere')
#
#     # VirtualFleet, get a deployment plan:
#     if not args.json:
#         puts("\nVirtualFleet, get a deployment plan...")
#     DF_PLAN = setup_deployment_plan(CENTER, THIS_DATE, nfloats=args.nfloats)
#     PLAN = {'lon': DF_PLAN['longitude'],
#             'lat': DF_PLAN['latitude'],
#             'time': np.array([np.datetime64(t) for t in DF_PLAN['date'].dt.strftime('%Y-%m-%d %H:%M').array]),
#             }
#     if not args.json:
#         puts("\t%i virtual floats to deploy" % DF_PLAN.shape[0], color=COLORS.green)
#
#     # Set up VirtualFleet:
#     if not args.json:
#         puts("\nVirtualFleet, set-up the fleet...")
#     VFleet = VirtualFleet(plan=PLAN,
#                           fieldset=VEL,
#                           mission=CFG)
#
#     # VirtualFleet, execute the simulation:
#     if not args.json:
#         puts("\nVirtualFleet, execute the simulation...")
#
#     # Remove traj file if exists:
#     output_path = os.path.join(WORKDIR, 'trajectories_%s.zarr' % get_sim_suffix(args, CFG))
#     # if os.path.exists(output_path):
#     #     shutil.rmtree(output_path)
#     #
#     # VFleet.simulate(duration=timedelta(hours=N_DAYS*24+1),
#     #                 step=timedelta(minutes=5),
#     #                 record=timedelta(minutes=30),
#     #                 output=True,
#     #                 output_folder=WORKDIR,
#     #                 output_file='trajectories_%s.zarr' % get_sim_suffix(args, CFG),
#     #                 verbose_progress=not args.json,
#     #                 )
#
#     # VirtualFleet, get simulated profiles index:
#     if not args.json:
#         puts("\nExtract swarm profiles index...")
#
#     T = Trajectories(WORKDIR + "/" + 'trajectories_%s.zarr' % get_sim_suffix(args, CFG))
#     DF_SIM = T.get_index().add_distances(origin=[THIS_PROFILE['longitude'].values[0], THIS_PROFILE['latitude'].values[0]])
#     if not args.json:
#         puts(str(T), color=COLORS.magenta)
#         puts(DF_SIM.head().to_string(), color=COLORS.green)
#     figure_positions(args, VEL, DF_SIM, DF_PLAN, THIS_PROFILE, CFG, WMO, CYC, VEL_NAME,
#                      dd=1, save_figure=args.save_figure, workdir=WORKDIR)
#
#     # Recovery, make predictions based on simulated profile density:
#     SP = SimPredictor(DF_SIM, THIS_PROFILE)
#     if not args.json:
#         puts("\nPredict float cycle position(s) from swarm simulation...", color=COLORS.white)
#         puts(str(SP), color=COLORS.magenta)
#     SP.fit_predict()
#     SP.add_metrics(VEL)
#     SP.plot_predictions(VEL,
#                          CFG,
#                          sim_suffix=get_sim_suffix(args, CFG),
#                          save_figure=args.save_figure,
#                          workdir=WORKDIR,
#                          orient='portrait')
#     results = SP.predictions
#
#     # Recovery, compute more swarm metrics:
#     for this_cyc in T.sim_cycles:
#         jsmetrics, fig, ax = T.analyse_pairwise_distances(cycle=this_cyc,
#                                                           save_figure=True,
#                                                           this_args=args,
#                                                           this_cfg=CFG,
#                                                           sim_suffix=get_sim_suffix(args, CFG),
#                                                           workdir=WORKDIR,
#                                                           )
#         if 'metrics' in results['predictions'][this_cyc]:
#             for key in jsmetrics.keys():
#                 results['predictions'][this_cyc]['metrics'].update({key: jsmetrics[key]})
#         else:
#             results['predictions'][this_cyc].update({'metrics': jsmetrics})
#
#     # Recovery, finalize JSON output:
#     execution_end = time.time()
#     process_end = time.process_time()
#     computation = {
#         'Date': pd.to_datetime('now', utc=True),
#         'Wall-time': pd.Timedelta(execution_end - execution_start, 's'),
#         'CPU-time': pd.Timedelta(process_end - process_start, 's'),
#         'system': getSystemInfo()
#     }
#     results['meta'] = {'Velocity field': VEL_NAME,
#                        'Nfloats': args.nfloats,
#                        'Computation': computation,
#                        'VFloats_config': CFG.to_json(),
#                        }
#
#     if not args.json:
#         puts("\nPredictions:")
#     results_js = json.dumps(results, indent=4, sort_keys=True, default=str)
#
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
