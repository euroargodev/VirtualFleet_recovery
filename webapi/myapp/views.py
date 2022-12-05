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
print("myapp/views.py:", app.config)
print(os.getcwd())

from .utils.flask import Args, parse_args, load_data_for, request_opts_for_data
from .utils.flask import read_params_from_path, search_local_prediction_datafiles, search_local_prediction_figfiles
from .utils.misc import strfdelta, get_traj
from .utils.html import Bootstrap_Carousel, Bootstrap_Figure, Bootstrap_Accordion


@app.route('/', defaults={'wmo': None, 'cyc': None}, methods=['GET'])
@app.route('/<int:wmo>', defaults={'cyc': None}, methods=['GET'])
@app.route('/<int:wmo>/<int:cyc>', methods=['GET'])
def index(wmo, cyc):
    # Parse request parameters:
    # wmo = wmo if wmo is not None else 0
    # cyc = cyc if cyc is not None else 0
    args = parse_args(wmo, cyc)
    print(args.amap)

    if args.wmo is not None and args.cyc is not None:
        return redirect(url_for('.results', **request_opts_for_data(request, args)))
    else:
        template_data = {'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
                         'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
                         'runs_html': None,  # get_html_of_simulations_accordion(args.output, request.base_url),
                         'app_url': request.url_root,
                         'url_form': url_for(".trigger", **request_opts_for_data(request, args)),
                         'css': url_for("static", filename="css")}
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

    carousel_html = Bootstrap_Carousel(slist, 'recapCarousel', args).html if len(slist) > 0 else None
    template_data = {'css': url_for("static", filename="css"),
                     'js': url_for("static", filename="js"),
                     'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
                     'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
                     'carousel_html': carousel_html,
                     'WMO': opts['wmo'],
                     'CYC': opts['cyc'],
                     'VELOCITY': htmlopts('velocity'),
                     'NFLOATS': htmlopts('nfloats'),
                     'CFG_PARKING_DEPTH': htmlopts('cfg_parking_depth'),
                     'CFG_CYCLE_DURATION': htmlopts('cfg_cycle_duration'),
                     'file_number': len(slist),
                     'app_url': request.url_root,
                     'url_recap': url_for(".recap", **opts),
                     'url_map': url_for(".map_error", **opts),
                     'ea_float': argopy.dashboard(argopy.utilities.check_wmo(args.wmo), url_only=True) if args.wmo is not None else None,
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

    df_float = argopy.utilities.get_coriolis_profile_id(wmo)
    # if cyc not in df_float['CYCLE_NUMBER']:

    # Init some variables used in template
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
        'url_data': url_for(".data", **args.amap),
        'url_predict': url_for(".predict", **args.amap),
        'url_form': url_for(".trigger", **args.amap),
        'url_recap': url_for(".recap", **opts),
        'url_map': url_for(".map_error", **opts),
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
        'url_previous': url_for(".results", **{**opts, **{'cyc': args.cyc-1}}),
        'url_next': url_for(".results", **{**opts, **{'cyc': args.cyc+1}}),
        'ea_profile': None,
        'vfloatcfg': None,
    }

    if args.cyc == 0:
        template_data['url_previous'] = None

    if args.cyc == df_float['CYCLE_NUMBER'].max():
        template_data['url_next'] = url_for(".trigger", **{**opts, **{'cyc': args.cyc+1}})

    # Load data for this set-up:
    # legacy = not 'cfg_parking_depth' in request.args
    # print('legacy', legacy)
    # jsdata = load_data_for(args, legacy=legacy)
    jsdata = load_data_for(args)
    print(jsdata)

    if jsdata is not None:
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
        template_data['figures'] = Bootstrap_Accordion(data=data, name='Figures').html

        template_data['prediction_lon'] = "%0.3f" % jsdata['prediction_location']['longitude']['value']
        template_data['prediction_lon_unit'] = "%s" % jsdata['prediction_location']['longitude']['unit']#.replace("degree", "deg")
        template_data['prediction_lat'] = "%0.3f" % jsdata['prediction_location']['latitude']['value']
        template_data['prediction_lat_unit'] = "%s" % jsdata['prediction_location']['latitude']['unit']#.replace("degree", "deg")
        template_data['prediction_time'] = "%s UTC" % jsdata['prediction_location']['time']['value']
        if 'score' in jsdata['prediction_metrics']['pairwise_distances']:
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

            template_data['ea_profile'] = jsdata['profile_to_predict']['url_profile']

        template_data['computation_walltime'] = strfdelta(pd.Timedelta(jsdata['meta']['Computation']['Wall-time']))
        template_data['computation_platform'] = "%s (%s)" % (jsdata['meta']['Computation']['system']['platform'],
                                                             jsdata['meta']['Computation']['system']['architecture'])
        if 'VFloats_config' in jsdata['meta']:
            template_data['vfloatcfg'] = jsdata['meta']['VFloats_config']

    template_data['ea_float'] = argopy.dashboard(argopy.utilities.check_wmo(args.wmo), url_only=True)

    html = render_template('results4.html', **template_data)
    return html


@app.route('/predict/<int:wmo>/<int:cyc>', methods=['GET', 'POST'])
def predict(wmo, cyc):
    """
    swagger_from_file: predict.yml
    """
    # Parse request parameters:
    args = parse_args(wmo, cyc)
    print(args.amap)
    print(request.args)

    # Load data for this set-up:
    jsdata = load_data_for(args)

    # If we didn't already use it, make a prediction:
    if jsdata is None:
        predictor(args)  # This can take a while...
        jsdata = load_data_for(args)

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
        jsdata = load_data_for(this_args, legacy=False)
        f = Feature(geometry=Point(
            (jsdata['prediction_location']['longitude']['value'], jsdata['prediction_location']['latitude']['value'])),
                    properties=jsdata)
        feature_list.append(f)

    # if len(feature_list) > 1000:
    #     feature_list = feature_list[0:1000]

    return jsonify(FeatureCollection(feature_list))


@app.route('/map_error', defaults={'wmo': None}, methods=['GET', 'POST'])
@app.route('/map_error/<int:wmo>', methods=['GET', 'POST'])
def map_error(wmo):
    # Parse request parameters:
    wmo = wmo if wmo is not None else 0
    args = parse_args(wmo, 0)
    # print('call to /map', args.amap)
    # print(url_for('data', wmo=args.wmo if args.wmo != 0 else None, nfloats=args.nfloats, velocity=args.velocity))

    template_data = {'css': url_for("static", filename="css"),
                     'js': url_for("static", filename="js"),
                     'dist': url_for("static", filename="dist"),
                     'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
                     'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
                     'url_app': request.url_root,
                     'url_recap': url_for(".recap", **request_opts_for_data(request, args)),
                     'url_map': url_for(".map_error", **request_opts_for_data(request, args)),
                     'url_wmomap': url_for(".map_error", **request_opts_for_data(request, args)),
                     'url_form': url_for(".trigger", **request_opts_for_data(request, args)),
                     'url_data': url_for('.data', **request_opts_for_data(request, args)),
                     'WMO': args.wmo if args.wmo != 0 else None,
                     'CYC': args.cyc if args.wmo != 0 else None,
                     'VELOCITY': args.velocity,
                     'NFLOATS': args.nfloats,
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
    args = parse_args(wmo, cyc)

    template_data = {'css': url_for("static", filename="css"),
                     'js': url_for("static", filename="js"),
                     'dist': url_for("static", filename="dist"),
                     'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
                     'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
                     'app_url': request.url_root,
                     'WMO': args.wmo if args.wmo != 0 else None,
                     'CYC': args.cyc if args.cyc != 0 else None,
                     'VELOCITY': args.velocity,
                     'NFLOATS': args.nfloats,
                     'CFG_PARKING_DEPTH': args.cfg_parking_depth,
                     'CFG_CYCLE_DURATION': args.cfg_cycle_duration,
                     'jsdata': url_for('.data', **request_opts_for_data(request, args)),
                     }

    if request.method == 'POST':

        float_cfg = {}
        if request.form['cfg_parking_depth'] == '':
            float_cfg['parking_depth'] = args.cfg_parking_depth
        else:
            float_cfg['parking_depth'] = int(request.form['cfg_parking_depth'])

        if request.form['cfg_cycle_duration'] == '':
            float_cfg['cycle_duration'] = args.cfg_cycle_duration
        else:
            float_cfg['cycle_duration'] = int(request.form['cfg_cycle_duration'])

        # Trigger prediction
        WMO = int(request.form['WMO'])
        CYC = int(request.form['CYC'])
        nfloats = int(request.form['nfloats'])
        velocity = request.form['VELOCITY']

        form_args = Args(WMO, CYC, json=True)
        form_args.nfloats = int(nfloats)
        form_args.velocity = velocity
        form_args.cfg_parking_depth = int(float_cfg['parking_depth'])
        form_args.cfg_cycle_duration = int(float_cfg['cycle_duration'])

        # print()
        print(request)
        print(args.amap)
        print(form_args.amap)

        # print(float_cfg)
        # url_predict = url_for(".predict", wmo=args.wmo, cyc=args.cyc, nfloats=args.nfloats, velocity=args.velocity)
        url_predict = url_for(".predict", **form_args.amap, redirect=True)
        url_results = url_for('.results', **form_args.amap)

        # Check if results are already available, otherwise, trigger prediction:
        if load_data_for(form_args) is not None:
            print('Found results, redirect to results page')
            return redirect(url_results)
        else:
            print('No results, trigger computation')
            return redirect(url_predict)

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
