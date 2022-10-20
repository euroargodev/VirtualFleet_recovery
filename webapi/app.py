#!/bin/env python
# -*coding: UTF-8 -*-
#
# export FLASK_DEBUG=True
# export FLASK_APP=app.py
# flask run
# flask run --host=134.246.146.178
#
# Created by gmaze on 19/10/2022
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
import sys
import json
from flask import Flask, request, url_for, redirect, jsonify
from flask import render_template
from flask_swagger import swagger

from markupsafe import escape

sys.path.insert(0, "../cli")
from recovery_prediction import predictor

app = Flask(__name__)


class Args:

    def __init__(self, wmo, cyc, *args, **kwargs):
        self.wmo = wmo
        self.cyc = cyc
        self.vf = None
        if 'nfloats' in kwargs:
            self.nfloats = kwargs['nfloats']
        else:
            self.nfloats = 2000

        if 'output' in kwargs:
            self.output = kwargs['output']
        else:
            self.output = './static/data'

        if 'velocity' in kwargs:
            self.velocity = kwargs['velocity']
        else:
            self.velocity = 'ARMOR3D'

        if 'save_figure' in kwargs:
            self.save_figure = kwargs['save_figure']
        else:
            self.save_figure = True

        if 'json' in kwargs:
            self.json = kwargs['json']
        else:
            self.json = False

    def __iter__(self):
        self.__i = 0
        self.__l = ['wmo', 'cyc', 'nfloats', 'velocity']
        return self

    def __next__(self):
        if self.__i < len(self.__l):
            i = self.__i
            self.__i += 1
            return self.__l[i]
        else:
            raise StopIteration()

    @property
    def amap(self):
        # [m.__setitem__(i, Args(6903576, 12).__getattribute__(i)) for i in Args(6903576, 12)]
        m = {'wmo': self.wmo, 'cyc': self.cyc, 'nfloats': self.nfloats, 'velocity': self.velocity}
        return m

    def html(self):
        summary = [""]
        summary.append("WMO: %i" % self.wmo)
        summary.append("CYC: %i" % self.cyc)
        summary.append("nfloats: %i" % self.nfloats)
        summary.append("velocity: %s" % self.velocity)
        summary.append("<hr>")
        summary.append("<b>VirtualFleet Recovery</b>")
        summary.append("(c) Argo-France/Ifremer/LOPS, 2022")
        return "<br>".join(summary)


def simulation_path(this_args):
    """Return relative local path to simulation folder
    Simulation path is determined using args
    """
    return os.path.sep.join(["data", str(this_args.wmo), str(this_args.cyc)])


def simulation_file_url(this_args, filename):
    """Return the URL toward a simulation file
    Simulation path is determined using args
    """
    this_path = os.path.sep.join([simulation_path(this_args), filename])
    # print("\n", filename, "\n", this_path, "\n", request.base_url, "\n", url_for('static', filename=this_path))
    url = "/".join([request.base_url, url_for('static', filename=this_path)])
    if 'predict/' in url:
        url = url.replace("predict/%i/%i//" % (this_args.wmo, this_args.cyc), "")
    elif 'results/' in url:
        url = url.replace("results/%i/%i//" % (this_args.wmo, this_args.cyc), "")
    return url


def complete_data_for(this_args, this_js):
    """Return API completed json data from a simulation
    Simulation parameters are determined using args
    """
    # Add url to figures in the json result:
    figlist = {'predictions': simulation_file_url(this_args, "vfrecov_predictions_%s_%i.png" % (this_args.velocity, this_args.nfloats)),
               # 'toto': simulation_file_url(this_args, "toto_%s_%i.png" % (this_args.velocity, this_args.nfloats)),
               'velocity': simulation_file_url(this_args, "vfrecov_velocity_%s.png" % (this_args.velocity)),
               'positions': simulation_file_url(this_args, "vfrecov_positions_%s_%i.png" % (this_args.velocity, this_args.nfloats))}
    this_js['meta']['figures'] = figlist
    return this_js


def load_data_for(this_args):
    """Return raw json file data from a simulation
    Simulation parameters are determined using args
    """
    js = os.path.sep.join(["data",
                           str(this_args.wmo),
                           str(this_args.cyc),
                           "prediction_%s_%i.json" % (this_args.velocity, this_args.nfloats)])
    ajs = os.path.abspath(os.path.sep.join([".", "static", js]))
    if os.path.exists(ajs):
        with open(ajs) as f:
            jsdata = json.load(f)
    else:
        jsdata = None

    if jsdata is not None:
        jsdata = complete_data_for(this_args, jsdata)

    return jsdata


def parse_args(wmo, cyc):
    """Return request parameters as an Args instance"""
    WMO = int(escape(wmo))
    CYC = int(escape(cyc))
    args = Args(WMO, CYC, json=True)
    args.nfloats = request.args.get('nfloats', args.__getattribute__('nfloats'), int)
    args.velocity = request.args.get('velocity', args.__getattribute__('velocity'), str)
    return args


@app.route('/')
def index():
    return 'VirtualFleet Recovery Server Works!'

# @app.route('/<wmo>/<cyc>', methods=['GET', 'POST'])
# def get_user(wmo, cyc):
#     s = f'WMO {escape(wmo)} CYC {escape(cyc)}'
#     return s


@app.route('/test/<int:wmo>/<int:cyc>', methods=['GET', 'POST'])
def test(wmo, cyc):
    WMO = int(escape(wmo))
    CYC = int(escape(cyc))
    args = Args(WMO, CYC)
    args.nfloats = request.args.get('nfloats', args.__getattribute__('nfloats'), int)
    args.velocity = request.args.get('velocity', args.__getattribute__('velocity'), str)
    return args.html()


@app.route('/predict/<int:wmo>/<int:cyc>', methods=['GET', 'POST'])
def predict(wmo, cyc):
    """
    swagger_from_file: predict.yml
    """
    # Parse request parameters:
    args = parse_args(wmo, cyc)

    # Load data for this set-up:
    jsdata = load_data_for(args)
    # If we didn't already use it, make a prediction:
    if jsdata is None:
        predictor(args)  # This can take a while...
        jsdata = load_data_for(args)

    return jsonify(jsdata)


@app.route('/results/<int:wmo>/<int:cyc>', methods=['GET'])
def results(wmo, cyc):
    # Parse request parameters:
    args = parse_args(wmo, cyc)

    # Init some variables used in template
    template_data = {
        'WMO': args.wmo,
        'CYC': args.cyc,
        'url_predict': url_for("predict", wmo=args.wmo, cyc=args.cyc, nfloats=args.nfloats, velocity=args.velocity),
        'prediction_src': None,
        'velocity_src': None,
        'data_js': None
    }

    # Load data for this set-up:
    jsdata = load_data_for(args)
    # print(jsdata)

    if jsdata is not None:
        template_data['prediction_src'] = jsdata['meta']['figures']['predictions']
        template_data['velocity_src'] = jsdata['meta']['figures']['velocity']
        template_data['data_js'] = url_for('predict', **args.amap)
        template_data['ea_float'] = jsdata['profile_to_predict']['url_float']
        template_data['ea_profile'] = jsdata['profile_to_predict']['url_profile']

    html = render_template('results.html', **template_data)
    return html


@app.route('/data/<int:wmo>/<int:cyc>', methods=['GET'])
def data(wmo, cyc):
    # Parse request parameters:
    args = parse_args(wmo, cyc)
    return redirect(url_for('predict', wmo=args.wmo, cyc=args.cyc, nfloats=args.nfloats, velocity=args.velocity))


# @app.route("/spec")
# def spec():
#     base_path = os.path.join(app.root_path, 'docs')
#     swag = swagger(app)
#     swag['info']['version'] = "1.0"
#     swag['info']['title'] = "API to the VirtualFleet Recovery Predictor"
#     return jsonify(swag)
#     # return jsonify(swag, from_file_keyword="swagger_from_file", base_path=base_path)


if __name__ == '__main__':
    app.run()