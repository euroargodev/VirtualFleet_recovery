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
import glob
from flask import Flask, request, url_for, redirect, jsonify
from flask import render_template
from flask_swagger import swagger

from markupsafe import escape

sys.path.insert(0, "../cli")
from recovery_prediction import predictor
import argopy

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

        if 'save_sim' in kwargs:
            self.save_sim = kwargs['save_sim']
        else:
            self.save_sim = True

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
    # print(url, 'predict/' in url, 'results/' in url)
    if 'predict/' in url:
        url = url.replace("predict/%i/%i//" % (this_args.wmo, this_args.cyc), "")
    elif 'results/' in url:
        # print("results/%i/%i//" % (this_args.wmo, this_args.cyc))
        url = url.replace("results/%i/%i//" % (this_args.wmo, this_args.cyc), "")
    # url = url.replace("//", "/")
    # print(url)
    return url


def complete_data_for(this_args, this_js):
    """Return API completed json data from a simulation
    Simulation parameters are determined using args
    """
    # Add url to figures in the json result:
    figlist = {'predictions': simulation_file_url(this_args, "vfrecov_predictions_%s_%i.png" % (this_args.velocity, this_args.nfloats)),
               'metrics': simulation_file_url(this_args, "vfrecov_metrics01_%s_%i.png" % (this_args.velocity, this_args.nfloats)),
               'velocity': simulation_file_url(this_args, "vfrecov_velocity_%s.png" % (this_args.velocity)),
               'positions': simulation_file_url(this_args, "vfrecov_positions_%s_%i.png" % (this_args.velocity, this_args.nfloats)),
               'predictions_recap': simulation_file_url(this_args, "vfrecov_predictions_recap_%s_%i.png" % (this_args.velocity, this_args.nfloats))}
    this_js['meta']['figures'] = figlist
    return this_js


def load_data_for(this_args):
    """Return the complete json file data from a simulation
    Simulation parameters are determined using args
    Raw data are complemented with results from complete_data_for() function
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


def get_html_of_simulations_list(this_src, this_urlroot):
    pattern = os.path.sep.join([this_src, "*", "*", "prediction_*.json"])
    # print(pattern)
    flist = sorted(glob.glob(pattern))
    if len(flist) == 0:
        return None
    WMOs = {}
    for f in flist:
        p = f.replace(this_src, "").split(os.path.sep)
        wmo, cyc, js = p[1], p[2], p[-1]
        wmo, cyc = int(wmo), int(cyc)
        cyc = "%.3d" % cyc
        # velocity, nfloats = js.replace("prediction_", "").replace(".json", "").split("_")[0], \
        #                     js.replace("prediction_", "").replace(".json", "").split("_")[1]
        # print(wmo, cyc, velocity, nfloats)
        if wmo not in WMOs:
            WMOs[wmo] = {}
        if cyc not in WMOs[wmo]:
            WMOs[wmo][cyc] = []
        WMOs[wmo][cyc].append(js)
    WMOs = dict(sorted(WMOs.items()))

    f_wline = "<li>\n<ul><h3>{wmo}</h3>\n{cycs}\n</ul>\n</li>".format
    f_cline = "<li><b>{cyc}:</b> {links}</li>".format
    f_html_link = "<a href=\"{url}\">{text}</a>".format
    f_app_url = "{root}/results/{wmo}/{cyc}".format

    lines = ["<ul>"]
    for wmo in WMOs:
        clines = []
        cyc_list = dict(sorted(WMOs[wmo].items()))
        for cyc in cyc_list:
            links = []
            for run in WMOs[wmo][cyc]:
                # links.append(f_html_link(url = os.path.sep.join([src, str(wmo), str(cyc), run]),
                #                          text = run.replace("prediction_","").replace(".json","").replace("_","-")))
                links.append(f_html_link(url=f_app_url(root=this_urlroot, wmo=wmo, cyc=cyc),
                                         text=run.replace("prediction_", "").replace(".json", "").replace("_", "-")))
            links = ", ".join(links)
            clines.append(f_cline(cyc=cyc, links=links))
        clines = "\n".join(clines)
        lines.append(f_wline(wmo=wmo, cycs=clines))
    lines.append("</ul>")
    return "\n".join(lines)


def get_html_of_simulations_accordion(this_src, this_urlroot):
    flist = sorted(glob.glob(os.path.sep.join([this_src, "*", "*", "prediction_*.json"])))
    WMOs = {}
    for f in flist:
        p = f.replace(this_src, "").split(os.path.sep)
        wmo, cyc, js = p[1], p[2], p[-1]
        wmo, cyc = int(wmo), int(cyc)
        cyc = "%.3d" % cyc
        # velocity, nfloats = js.replace("prediction_", "").replace(".json", "").split("_")[0], \
        #                     js.replace("prediction_", "").replace(".json", "").split("_")[1]
        # print(wmo, cyc, velocity, nfloats)
        if wmo not in WMOs:
            WMOs[wmo] = {}
        if cyc not in WMOs[wmo]:
            WMOs[wmo][cyc] = []
        WMOs[wmo][cyc].append(js)
    WMOs = dict(sorted(WMOs.items()))

    f_accordionItem = "<div class=\"accordion-item\">\
    <h2 class=\"accordion-header\" id=\"{wmo}\">\
    <button class=\"accordion-button {collapsed}\" type=\"button\" data-bs-toggle=\"collapse\" data-bs-target=\"#collapse{wmo}\" aria-expanded=\"true\" aria-controls=\"collapse{wmo}\">\
        Float {wmo} ({ncyc} cycles simulated)\
    </button>\
    </h2>\
    <div id=\"collapse{wmo}\" class=\"accordion-collapse collapse {show}\" aria-labelledby=\"{wmo}\" data-bs-parent=\"#accordionSimulations\">\
        <div class=\"accordion-body\">\
            {cycs}\
        </div>\
    </div>\
    </div>".format

    # f_wline = "<li>\n<ul><h3>{wmo}</h3>\n{cycs}\n</ul>\n</li>".format
    f_cline = "<li><b>{cyc}:</b> {links}</li>".format
    f_html_link = "<a href=\"{url}\" target=\"blank\">{text}</a>".format
    # f_app_url = "{root}results/{wmo}/{cyc}".format
    f_app_url = "{root}results/{wmo}/{cyc}?velocity={velocity}&nfloats={nfloats}".format

    lines = ["<div class=\"accordion\" id=\"accordionSimulations\">"]
    for iw, wmo in enumerate(WMOs):
        clines = []
        cyc_list = dict(sorted(WMOs[wmo].items()))
        for cyc in cyc_list:
            links = []
            for run in WMOs[wmo][cyc]:
                label = run.replace("prediction_", "").replace(".json", "")
                # url = f_app_url(root=this_urlroot, wmo=wmo, cyc=int(cyc))
                velocity, nfloats = label.split("_")[0], label.split("_")[1]
                url = f_app_url(root=this_urlroot, wmo=wmo, cyc=int(cyc), velocity=velocity, nfloats=nfloats)
                links.append(f_html_link(url=url, text=label.replace("_", "/N=")))
            links = ", ".join(links)
            clines.append(f_cline(cyc=cyc, links=links))
        clines = "".join(clines)
        # lines.append(f_wline(wmo=wmo, cycs=clines))
        if iw < 0:
            show = 'show'
            collapsed = ''
        else:
            show = ''
            collapsed = 'collapsed'
        lines.append(f_accordionItem(wmo=wmo, cycs=clines, show=show, collapsed=collapsed, ncyc=len(cyc_list)))
    lines.append("</div>")
    return "\n".join(lines)


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
    # Parse request parameters:
    # (none in this case, we just need the `args` object)
    args = parse_args(0, 0)

    template_data = {'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
                     'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
                     'runs_html': get_html_of_simulations_accordion(args.output, request.base_url)}
    # print(template_data['runs_html'])

    html = render_template('index.html', **template_data)
    return html


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
        'css': url_for("static", filename="css"),
        'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
        'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
        'WMO': args.wmo,
        'CYC': args.cyc,
        'url': request.base_url,
        'url_predict': url_for("predict", wmo=args.wmo, cyc=args.cyc, nfloats=args.nfloats, velocity=args.velocity),
        'prediction_src': None,
        'metric_src': None,
        'velocity_src': None,
        'prediction_recap_src': None,
        'data_js': None,
        'prediction_lon': None,
        'prediction_lat': None,
        'error_bearing': None,
        'error_dist': None,
    }

    # Load data for this set-up:
    jsdata = load_data_for(args)
    # print(jsdata)

    if jsdata is not None:
        template_data['prediction_src'] = jsdata['meta']['figures']['predictions']
        template_data['prediction_recap_src'] = jsdata['meta']['figures']['predictions_recap']
        template_data['velocity_src'] = jsdata['meta']['figures']['velocity']
        template_data['metric_src'] = jsdata['meta']['figures']['metrics']
        template_data['data_js'] = url_for('predict', **args.amap)
        template_data['prediction_lon'] = "%0.4f" % jsdata['prediction_location']['longitude']['value']
        template_data['prediction_lon_unit'] = "%s" % jsdata['prediction_location']['longitude']['unit']
        template_data['prediction_lat'] = "%0.4f" % jsdata['prediction_location']['latitude']['value']
        template_data['prediction_lat_unit'] = "%s" % jsdata['prediction_location']['latitude']['unit']

        template_data['error_bearing'] = "%0.1f" % jsdata['prediction_location_error']['bearing']['value']
        template_data['error_bearing_unit'] = "%s" % jsdata['prediction_location_error']['bearing']['unit']
        template_data['error_dist'] = "%0.1f" % jsdata['prediction_location_error']['distance']['value']
        template_data['error_dist_unit'] = "%s" % jsdata['prediction_location_error']['distance']['unit']

        # template_data['ea_float'] = jsdata['profile_to_predict']['url_float']
        # template_data['ea_profile'] = jsdata['profile_to_predict']['url_profile']

    template_data['ea_float'] = argopy.dashboard(argopy.utilities.check_wmo(args.wmo), url_only=True)
    template_data['ea_profile'] = argopy.dashboard(argopy.utilities.check_wmo(args.wmo), argopy.utilities.check_cyc(args.cyc), url_only=True)

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
