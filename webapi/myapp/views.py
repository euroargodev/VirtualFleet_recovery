#!/bin/env python
# -*coding: UTF-8 -*-
#
# export FLASK_DEBUG=True
# export FLASK_APP=app.py
# flask run
# flask run --host=134.246.146.178 # Laptop
# flask run --host=134.246.146.54 # Pacnet
#
#
"""

Make a prediction for the position of the CYC cycle from float WMO
(this will return a json file with the prediction results):
    http://134.246.146.178:5000/predict/<WMO>/<CYC>
Possible options:
    http://134.246.146.178:5000/predict/<WMO>/<CYC>?nfloats=1000
    http://134.246.146.178:5000/predict/<WMO>/<CYC>?velocity=ARMOR3D
    http://134.246.146.178:5000/predict/<WMO>/<CYC>?velocity=GLORYS

Get a webpage with figures or link to make predictions:
    http://134.246.146.178:5000/results/<WMO>/<CYC>

"""

import os
import pandas as pd
import sys
import glob
from flask import Flask, request, url_for, redirect, jsonify
from flask import render_template
# from flask_swagger import swagger

# from markupsafe import escape
import argopy

sys.path.insert(0, "../cli")
from recovery_prediction import predictor

from geojson import Feature, Point, FeatureCollection, LineString
# from string import Formatter

from myapp import app
APP_NAME = __name__.split('.')[0]
# print("myapp/views.py:", app.config)
# print(os.getcwd())

from .utils.for_flask import Args, parse_args, get_sim_files, load_data_for, request_opts_for_data
from .utils.for_flask import read_params_from_path, search_local_prediction_datafiles, search_local_prediction_figfiles
from .utils.misc import strfdelta, get_traj
from .utils.bootstrap import Bootstrap_Figure, Bootstrap_Accordion
from .utils.html import Bootstrap_Carousel_Recovery, get_the_sidebar


@app.route('/', defaults={'wmo': None, 'cyc': None}, methods=['GET'])
@app.route('/<int:wmo>', defaults={'cyc': None}, methods=['GET'])
@app.route('/<int:wmo>/<int:cyc>', methods=['GET'])
def index(wmo, cyc):
    # Parse request parameters:
    # wmo = wmo if wmo is not None else 0
    # cyc = cyc if cyc is not None else 0
    args = parse_args(wmo, cyc)
    # print(args.amap)

    if args.wmo is not None and args.cyc is not None:
        return redirect(url_for('.results', **request_opts_for_data(request, args)))
    else:
        template_data = {'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
                         'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
                         'runs_html': None,  # get_html_of_simulations_accordion(args.output, request.base_url),
                         'app_url': request.url_root,
                         'url_form': url_for(".trigger", **request_opts_for_data(request, args)),
                         'css': url_for("static", filename="css"),
                         'ea_profile': None,
                         'ea_float': None,
                         }
        # print(template_data['runs_html'])

        html = render_template('index2.html', **template_data)
        return html


@app.route('/recap', defaults={'wmo': None}, methods=['GET'])
@app.route('/recap/<int:wmo>', methods=['GET'])
def recap(wmo):
    # Parse request parameters:
    args = parse_args(wmo, None)
    opts = request_opts_for_data(request, args)
    htmlopts = lambda x: opts[x] if opts[x] is not None else '<any>'

    slist = search_local_prediction_figfiles(args, request)
    html_sidebar = get_the_sidebar(args, opts, None, active="Swipe all cycles")
    html_carousel = Bootstrap_Carousel_Recovery(slist, 'recapCarousel').html if len(slist) > 0 else None
    template_data = {'css': url_for("static", filename="css"),
                     'js': url_for("static", filename="js"),
                     'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
                     'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
                     'carousel_html': html_carousel,
                     'WMO': opts['wmo'],
                     'CYC': opts['cyc'],
                     'VELOCITY': htmlopts('velocity'),
                     'NFLOATS': htmlopts('nfloats'),
                     'CFG_PARKING_DEPTH': htmlopts('cfg_parking_depth'),
                     'CFG_CYCLE_DURATION': htmlopts('cfg_cycle_duration'),
                     'file_number': len(slist),
                     'app_url': request.url_root,
                     'sidebar': html_sidebar,
                     }

    html = render_template('list2.html', **template_data)
    return html


@app.route('/results/<int:wmo>/<int:cyc>', methods=['GET'])
def results(wmo, cyc):
    # Parse request parameters:
    args = parse_args(wmo, cyc)
    opts = request_opts_for_data(request, args)
    opts.pop('cyc')
    # print(args.amap)

    # Load data for this set-up:
    results = get_sim_files(args)
    if len(results) > 1:
        print("Found %i simulation(s) config for this profile, loading data for the 1st one" % len(results))
    jsdata, args = load_data_for(args)  # args is updated with data from the selected simulation
    # print(args.amap)

    # Load float trajectory:
    df_float = argopy.utilities.get_coriolis_profile_id(wmo)

    # Init some variables used in template
    html_sidebar = get_the_sidebar(args, opts, jsdata)

    template_data = {
        'css': url_for("static", filename="css"),
        'js': url_for("static", filename="js"),
        'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
        'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
        'WMO': args.wmo,
        'CYC': args.cyc,
        'VELOCITY': args.velocity,
        'NFLOATS': args.nfloats,
        'CFG_PARKING_DEPTH': args.cfg_parking_depth,
        'CFG_CYCLE_DURATION': args.cfg_cycle_duration,
        'url': request.base_url,
        'sidebar': html_sidebar,
        'url_previous': url_for(".results", **{**opts, **{'cyc': args.cyc-1}}),
        'url_next': url_for(".results", **{**opts, **{'cyc': args.cyc+1}}),
        'prediction_src': None,
        'metric_src': None,
        'velocity_src': None,
        'prediction_recap_src': None,
        'prediction_lon': None,
        'prediction_lat': None,
        'prediction_time': None,
        'prediction_score': None,
        'error_transit': None,
        'error_bearing': None,
        'error_dist': None,
        'vfloatcfg': None,
    }

    if args.cyc == 0 or args.cyc is None:
        template_data['url_previous'] = None

    if args.cyc == df_float['CYCLE_NUMBER'].max():
        template_data['url_next'] = url_for(".trigger", **{**opts, **{'cyc': args.cyc+1}})

    if jsdata is None:
        url_trigger = url_for(".trigger", **args.amap)
        return redirect(url_trigger)
    else:
        # print(jsdata['meta']['figures'])
        data = [
        {'title': 'Prediction',
         'body': Bootstrap_Figure(src=jsdata['meta']['figures']['predictions_recap']).html},
        {'title': 'Probabilistic prediction details',
         'body': Bootstrap_Figure(src=jsdata['meta']['figures']['predictions']).html},
        {'title': 'Trajectory analysis details',
         'body': Bootstrap_Figure(src=jsdata['meta']['figures']['metrics']).html},
        {'title': 'Velocity field domain',
         'body': Bootstrap_Figure(src=jsdata['meta']['figures']['velocity']).html},
            ]
        template_data['figures'] = Bootstrap_Accordion(data=data, id='Figures').html

        template_data['prediction_lon'] = "%0.3f" % jsdata['prediction_location']['longitude']['value']
        template_data['prediction_lon_unit'] = "%s" % jsdata['prediction_location']['longitude']['unit']#.replace("degree", "deg")
        template_data['prediction_lat'] = "%0.3f" % jsdata['prediction_location']['latitude']['value']
        template_data['prediction_lat_unit'] = "%s" % jsdata['prediction_location']['latitude']['unit']#.replace("degree", "deg")
        template_data['prediction_time'] = "%s UTC" % jsdata['prediction_location']['time']['value']
        if 'pairwise_distances' in jsdata['prediction_metrics'] and 'score' in jsdata['prediction_metrics']['pairwise_distances']:
            template_data['prediction_score'] = "%0.0f%%" % (100*float(jsdata['prediction_metrics']['pairwise_distances']['score']['value']))
            # template_data['prediction_score'] = "%0.0f%%" % (100*float(jsdata['prediction_metrics']['pairwise_distances']['overlapping']['value']))
        else:
            template_data['prediction_score'] = "?"

        if jsdata['prediction_location_error']['bearing']['value'] is not None:
            template_data['error_bearing'] = "%0.1f" % jsdata['prediction_location_error']['bearing']['value']
            template_data['error_bearing_unit'] = "%s" % jsdata['prediction_location_error']['bearing']['unit']
            template_data['error_dist'] = "%0.1f" % jsdata['prediction_location_error']['distance']['value']
            template_data['error_dist_unit'] = "%s" % jsdata['prediction_location_error']['distance']['unit']
            template_data['error_time'] = strfdelta(pd.Timedelta(float(jsdata['prediction_location_error']['time']['value']), unit='h'))
            # template_data['error_time_unit'] = "%s" % jsdata['prediction_location_error']['time']['unit']
            template_data['error_transit'] = strfdelta(pd.Timedelta(float(jsdata['prediction_metrics']['transit']['value']), unit='h'))

        template_data['computation_walltime'] = strfdelta(pd.Timedelta(jsdata['meta']['Computation']['Wall-time']))
        template_data['computation_platform'] = "%s (%s)" % (jsdata['meta']['Computation']['system']['platform'],
                                                             jsdata['meta']['Computation']['system']['architecture'])
        if 'VFloats_config' in jsdata['meta']:
            template_data['vfloatcfg'] = jsdata['meta']['VFloats_config']

        html = render_template('results4.html', **template_data)
        return html


@app.route('/predict/<int:wmo>/<int:cyc>', methods=['GET', 'POST'])
def predict(wmo, cyc):
    """
    swagger_from_file: predict.yml
    """
    # Parse request parameters:
    args = parse_args(wmo, cyc, default=False)
    # print("predict.args:", args.amap)
    # print(request.args)

    # Load data for this set-up:
    jsdata, _ = load_data_for(args)

    # If we didn't already use it, make a prediction:
    if jsdata is None:
        predictor(args)  # This can take a while...
        jsdata, _ = load_data_for(args)

    if 'redirect' in request.args:
        return redirect(url_for('.results', **args.amap))
    else:
        return jsonify(jsdata)


@app.route('/data', defaults={'wmo': None}, methods=['GET', 'POST'])
@app.route('/data/<int:wmo>', methods=['GET', 'POST'])
def data(wmo):
    # Parse request parameters:
    wmo = wmo if wmo is not None else 0
    args = parse_args(wmo, None)
    # args.cyc = args.cyc if args.cyc is not None else None
    # print('call to data/', args.amap)

    slist = search_local_prediction_datafiles(args)

    feature_list = []
    for filename in slist:
        this_wmo = filename.split(os.path.sep)[-3]
        this_cyc = filename.split(os.path.sep)[-2]
        params = read_params_from_path(filename, plist=['VEL', 'NF', 'CYCDUR', 'PDPTH'])
        opts = {'velocity': params['VEL'],
                'nfloats': int(params['NF']),
                'cfg_parking_depth': int(params['PDPTH']),
                'cfg_cycle_duration': int(params['CYCDUR'])}
        this_args = Args(this_wmo, this_cyc, **opts)
        jsdata, this_args = load_data_for(this_args, legacy=False)
        f = Feature(geometry=Point(
            (jsdata['prediction_location']['longitude']['value'], jsdata['prediction_location']['latitude']['value'])),
                    properties=jsdata)
        feature_list.append(f)

    # if len(feature_list) > 1000:
    #     feature_list = feature_list[0:1000]

    return jsonify(FeatureCollection(feature_list))


@app.route('/map_error', defaults={'wmo': None}, methods=['GET'])
@app.route('/map_error/<int:wmo>', methods=['GET'])
def map_error(wmo):
    # Parse request parameters:
    wmo = wmo if wmo is not None else 0
    args = parse_args(wmo, 0)
    # print('call to /map', args.amap)
    # print(url_for('data', wmo=args.wmo if args.wmo != 0 else None, nfloats=args.nfloats, velocity=args.velocity))
    opts = request_opts_for_data(request, args)
    html_sidebar = get_the_sidebar(args, opts, None, active="See on a map")
    template_data = {'css': url_for("static", filename="css"),
                     'js': url_for("static", filename="js"),
                     'dist': url_for("static", filename="dist"),
                     'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
                     'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
                     'sidebar': html_sidebar,
                     'url_app': request.url_root,
                     'url_wmomap': url_for(".map_error", **opts),
                     'url_data': url_for('.data', **opts),
                     'WMO': args.wmo if args.wmo != 0 else None,
                     'CYC': args.cyc if args.wmo != 0 else None,
                     'VELOCITY': args.velocity,
                     'NFLOATS': args.nfloats,
                     'CFG_PARKING_DEPTH': args.cfg_parking_depth,
                     'CFG_CYCLE_DURATION': args.cfg_cycle_duration,
                     'trajdata': get_traj(args.wmo if args.wmo != 0 else None),
                     }
    # print(jsonify(template_data))

    html = render_template('map_error.html', **template_data)
    return html


@app.route('/trigger', defaults={'wmo': None, 'cyc': None}, methods=['GET', 'POST'])
@app.route('/trigger/<int:wmo>', defaults={'cyc': None}, methods=['GET', 'POST'])
@app.route('/trigger/<int:wmo>/<int:cyc>', methods=['GET', 'POST'])
def trigger(wmo, cyc):
    # Parse request parameters:
    wmo = wmo if wmo is not None else 0
    cyc = cyc if cyc is not None else 0
    args = parse_args(wmo, cyc, default=False)
    # print("trigger.args:", args.amap)

    opts = request_opts_for_data(request, args)
    opts.pop('cyc')
    html_sidebar = get_the_sidebar(args, opts, None, active="Prediction form")
    template_data = {'css': url_for("static", filename="css"),
                     'js': url_for("static", filename="js"),
                     'dist': url_for("static", filename="dist"),
                     'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
                     'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
                     'app_url': request.url_root,
                     'WMO': args.wmo if args.wmo != 0 else None,
                     'CYC': args.cyc if args.cyc != 0 else None,
                     'VELOCITY': args.velocity if args.velocity is not None else app.config['DEFAULT_PARAMS']['velocity'],
                     'NFLOATS': args.nfloats if args.nfloats is not None else app.config['DEFAULT_PARAMS']['nfloats'],
                     'CFG_PARKING_DEPTH': args.cfg_parking_depth,
                     'CFG_CYCLE_DURATION': args.cfg_cycle_duration,
                     'jsdata': url_for('.data', **request_opts_for_data(request, args)),
                     'sidebar': html_sidebar,
                     'url_previous': url_for(".results", **{**opts, **{'cyc': args.cyc-1}}) if args.cyc != 0 else None,
                     'url_next': url_for(".results", **{**opts, **{'cyc': args.cyc+1}}) if args.cyc != 0 else None,
                     }

    if request.method == 'POST':  # We arrive here when the form submit button is pressed

        # Get prediction parameters from the submitted form:
        WMO = int(request.form['WMO'])
        CYC = int(request.form['CYC'])
        nfloats = int(request.form['nfloats'])
        velocity = request.form['VELOCITY']

        # float_cfg = {}
        # if request.form['cfg_parking_depth'] == '':
        #     float_cfg['parking_depth'] = args.cfg_parking_depth
        # else:
        #     float_cfg['parking_depth'] = int(request.form['cfg_parking_depth'])
        #
        # if request.form['cfg_cycle_duration'] == '':
        #     float_cfg['cycle_duration'] = args.cfg_cycle_duration
        # else:
        #     float_cfg['cycle_duration'] = int(request.form['cfg_cycle_duration'])

        form_args = Args(WMO, CYC, json=True)
        form_args.nfloats = int(nfloats)
        form_args.velocity = velocity
        form_args.cfg_parking_depth = None if request.form['cfg_parking_depth'] == '' else int(request.form['cfg_parking_depth'])
        form_args.cfg_cycle_duration = None if request.form['cfg_cycle_duration'] == '' else int(request.form['cfg_cycle_duration'])

        # print("trigger.request.form", request.form)
        # print(args.amap)
        # print("trigger.processed.to_predict:", form_args.amap)
        # print(float_cfg)

        # Check if results are already available, otherwise, trigger prediction:
        if load_data_for(form_args)[0] is not None:
            print('Found results, redirect to results page')
            url_results = url_for('.results', **form_args.amap)
            return redirect(url_results)
        else:
            print('No results, trigger computation')
            url_predict = url_for(".predict", **form_args.amap, redirect=True)
            return redirect(url_predict)
            # return jsonify(form_args.amap)
    else:
        html = render_template('trigger.html', **template_data)
        return html


# @app.route("/spec")
# def spec():
#     base_path = os.path.join(app.root_path, 'docs')
#     swag = swagger(app)
#     swag['info']['version'] = "1.0"
#     swag['info']['title'] = "API to the VirtualFleet Recovery Predictor"
#     return jsonify(swag)
#     # return jsonify(swag, from_file_keyword="swagger_from_file", base_path=base_path)
