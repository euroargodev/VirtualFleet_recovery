
def predict_function(
        wmo: int,
        cyc: int,
        n_predictions: int = 1,
):
    """
    Execute VirtualFleet-Recovery predictor and return results as a JSON string

    Parameters
    ----------
    wmo
    cyc
    n_predictions

    Returns
    -------
    data

    """  # noqa
    return {'wmo': wmo, 'cyc': cyc}


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

