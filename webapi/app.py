#!/bin/env python
# -*coding: UTF-8 -*-
#
# export FLASK_DEBUG=True
# export FLASK_APP=app.py
# flask run
# flask run --host=134.246.146.178 # Laptop
# flask run --host=134.246.146.54 # Pacnet
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
import pandas as pd
import sys
import json
import glob
import numpy as np
from flask import Flask, request, url_for, redirect, jsonify
from flask import render_template
from flask_swagger import swagger

from markupsafe import escape
import argopy

sys.path.insert(0, "../cli")
from recovery_prediction import predictor

from geojson import Feature, Point, FeatureCollection
from string import Formatter


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
        # summary.append("Simulation dashboard: %s" % "" )
        summary.append("<hr>")
        summary.append("<b>VirtualFleet Recovery</b>")
        summary.append("(c) Argo-France/Ifremer/LOPS, 2022")
        return "<br>".join(summary)


def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s', inputtype='timedelta'):
    """Convert a datetime.timedelta object or a regular number to a custom-
    formatted string, just like the stftime() method does for datetime.datetime
    objects.

    The fmt argument allows custom formatting to be specified.  Fields can
    include seconds, minutes, hours, days, and weeks.  Each field is optional.

    Some examples:
        '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
        '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
        '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
        '{H}h {S}s'                       --> '72h 800s'

    The inputtype argument allows tdelta to be a regular number instead of the
    default, which is a datetime.timedelta object.  Valid inputtype strings:
        's', 'seconds',
        'm', 'minutes',
        'h', 'hours',
        'd', 'days',
        'w', 'weeks'
    """

    # Convert tdelta to integer seconds.
    if inputtype == 'timedelta':
        remainder = int(tdelta.total_seconds())
    elif inputtype in ['s', 'seconds']:
        remainder = int(tdelta)
    elif inputtype in ['m', 'minutes']:
        remainder = int(tdelta)*60
    elif inputtype in ['h', 'hours']:
        remainder = int(tdelta)*3600
    elif inputtype in ['d', 'days']:
        remainder = int(tdelta)*86400
    elif inputtype in ['w', 'weeks']:
        remainder = int(tdelta)*604800

    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)


def simulation_path(this_args):
    """Return relative local path to simulation folder
    Simulation path is determined using args
    """
    return os.path.sep.join(["data", str(this_args.wmo), str(this_args.cyc)])


def simulation_file_url(this_args, filename, safe=False):
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
    elif 'test/' in url:
        url = url.replace("test/%i/%i//" % (this_args.wmo, this_args.cyc), "")
    elif 'data/' in url:
        url = url.replace("data//static/data/", "static/data/")
    # url = url.replace("//", "/")
    # print(url)
    if safe:
        local_file = os.path.sep.join([this_args.output, str(this_args.wmo), str(this_args.cyc), filename])
        local_file = os.path.abspath(local_file)
        if os.path.lexists(local_file):
            return url
        else:
            print("%s not found" % local_file)
            return None
    return url


def complete_data_for(this_args, this_js):
    """Return API completed json data from a simulation
    Simulation parameters are determined using args
    """
    # Add url to figures in the json result:
    figlist = {'predictions': simulation_file_url(this_args, "vfrecov_predictions_%s_%i.png" % (this_args.velocity, this_args.nfloats), safe=True),
               'metrics': simulation_file_url(this_args, "vfrecov_metrics01_%s_%i.png" % (this_args.velocity, this_args.nfloats), safe=True),
               'velocity': simulation_file_url(this_args, "vfrecov_velocity_%s.png" % (this_args.velocity)),
               'positions': simulation_file_url(this_args, "vfrecov_positions_%s_%i.png" % (this_args.velocity, this_args.nfloats), safe=True),
               'predictions_recap': simulation_file_url(this_args, "vfrecov_predictions_recap_%s_%i.png" % (this_args.velocity, this_args.nfloats), safe=True)}
    this_js['meta']['figures'] = figlist
    this_js['meta']['api'] = {'cycle_page': "".join([request.host_url[0:-1], url_for("index", wmo=this_args.wmo, cyc=this_args.cyc, nfloats=this_args.nfloats, velocity=this_args.velocity)]),
                              'float_page': "".join([request.host_url[0:-1], url_for("recap", wmo=this_args.wmo, nfloats=this_args.nfloats, velocity=this_args.velocity)]),
                              'float_map': "".join([request.host_url[0:-1], url_for("map", wmo=this_args.wmo, nfloats=this_args.nfloats, velocity=this_args.velocity)])}

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

    lines = ["<div class=\"accordion w-100\" id=\"accordionSimulations\">"]
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


class HtmlHelper:
    def __init__(self, indent=0):
        """HTML string formatting helper

        >>> HtmlHelper().cblock("p", content="Hello !", attrs={"class": "toto", "aria-hidden": "false"})
        '<p class="toto" aria-hidden="false">Hello !</p>'

        >>> HtmlHelper().block("img", attrs={"src": "fig.png"})
        '<img src="fig.png">'

        """
        self.indent = indent

    def __indent(self, txt):
        shift = " " * self.indent
        return "%s%s" % (shift, txt)

    def cblock(self, name, attrs={}, content=''):
        if len(attrs) > 0:
            html = "<%s %s>%s</%s>" % (
            name, " ".join(["%s=\"%s\"" % (key, attrs[key]) for key in attrs.keys() if attrs[key] != ""]), content,
            name)
        else:
            html = "<%s>%s</%s>" % (name, content, name)
        return self.__indent(html)

    def block(self, name, attrs={}):
        if len(attrs) > 0:
            html = "<%s %s>" % (
            name, " ".join(["%s=\"%s\"" % (key, attrs[key]) for key in attrs.keys() if attrs[key] != ""]))
        else:
            html = "<%s>" % name
        return self.__indent(html)


class Bootstrap_Carousel:

    def __init__(self, figure_list=[], name='carouselExample', args=None):
        """Create a Bootstrap Carousel for a given list of figure files"""
        self.flist = figure_list
        self.name = name
        self.args = args

    def __repr__(self):
        summary = []
        summary.append("<bootstrap.carouselWithCaption>")
        summary.append("Figures: %i" % len(self.flist))
        summary.append("Name: %s" % self.name)
        return "\n".join(summary)

    def __html_carousel_btn(self, islide=0, active=False, target='carouselExample'):
        attrs = {'type': "button",
                 'data-bs-target': "#%s" % target,
                 'data-bs-slide-to': "%i" % islide,
                 'aria-label': "Slide %i" % int(islide + 1),
                 'class': "active" if active else "",
                 }
        return HtmlHelper().cblock('button', attrs=attrs)

    def __get_list_of_carousel_btn_html(self, this_flist, carouselName='carouselExample'):
        html = []
        for islide in np.arange(0, len(this_flist)):
            html.append(self.__html_carousel_btn(islide, active=islide == 0, target=carouselName))
        html = "\n".join(html)
        return html

    def __html_carousel_item(self, src='...', label='Slide label',
                             description='Some representative placeholder content for the second slide.', active=False):
        html = []
        BH = lambda n: HtmlHelper(indent=n)
        html.append(BH(n=0).block("div",
                                  attrs={"class": "carousel-item %s" % ("active" if active else ""),
                                         "data-bs-interval": 10}))
        html.append(BH(n=2).block("img", attrs={'src': "{src}", 'class': 'd-block w-100', 'alt':''}))
        html.append(BH(n=2).block("div", attrs={'class': 'carousel-caption d-none d-md-block'}))
        html.append(BH(n=4).cblock("h5", content='{label}'))
        html.append(BH(n=4).cblock("p", content='{description}'))
        html.append(BH(n=2).block("/div"))
        html.append(BH(n=0).block("/div"))
        html = "\n".join(html).format(src=src, label=label, description=description)
        return html

    def __get_list_of_carousel_items_html(self, this_flist):
        html = []
        for islide, src in enumerate(this_flist):
            wmo = src.replace("static/data/", "").split("/")[1]
            cyc = src.replace("static/data/", "").split("/")[2]
            label = "Float %s - Cycle %s" % (wmo, cyc)
            results_lnk = "/".join([request.url_root, 'results', wmo, cyc]).replace("//results", "/results")
            results_lnk = "%s?velocity=%s&nfloats=%s" % (results_lnk, self.args.velocity, self.args.nfloats)
            description = HtmlHelper().cblock("a", attrs={"href": results_lnk, "target": ""},
                                              content="Check this cycle details")
            subsample_lnk = "/".join([request.url_root, 'recap', wmo]).replace("//recap", "/recap")
            subsample_lnk = "%s?velocity=%s&nfloats=%s" % (subsample_lnk, self.args.velocity, self.args.nfloats)
            description = "%s / %s" % (HtmlHelper().cblock("a", attrs={"href": subsample_lnk, "target": ""},
                                          content="Swipe only this float"), description)

            html.append(
                self.__html_carousel_item(src=src, label=label, description=description, active=islide == 0))
        html = "\n".join(html)
        return html

    @property
    def html(self):
        html = []
        BH = lambda n: HtmlHelper(indent=n)

        html.append(BH(n=0).block("div", attrs={"id": self.name,
                                                "class": "carousel carousel-dark slide",
                                                "data-bs-ride": "false"}))

        html.append(BH(n=2).block("div", attrs={"class": "carousel-indicators"}))
        html.append(self.__get_list_of_carousel_btn_html(self.flist, carouselName=self.name))
        html.append(BH(n=2).block("/div"))

        html.append(BH(n=2).block("div", attrs={"class": "carousel-inner"}))
        html.append(self.__get_list_of_carousel_items_html(self.flist))
        html.append(BH(n=2).block("/div"))

        html.append(BH(n=2).block("button", attrs={"type": "button",
                                                    "class": "carousel-control-prev",
                                                    "data-bs-target": "#%s" % self.name,
                                                    "data-bs-slide": "prev"}))
        html.append(BH(n=4).cblock("span", attrs={"class": "carousel-control-prev-icon",
                                                  "aria-hidden": "true"}))
        html.append(BH(n=4).cblock("span", attrs={"class": "visually-hidden"}, content="Previous"))
        html.append(BH(n=2).block("/button"))

        html.append(BH(n=2).block("button", attrs={"type": "button",
                                                    "class": "carousel-control-next",
                                                    "data-bs-target": "#%s" % self.name,
                                                    "data-bs-slide": "next"}))
        html.append(BH(n=4).cblock("span", attrs={"class": "carousel-control-next-icon",
                                                  "aria-hidden": "true"}))
        html.append(BH(n=4).cblock("span", attrs={"class": "visually-hidden"}, content="Next"))
        html.append(BH(n=2).block("/button"))

        html.append(BH(n=0).block("/div"))
        return "\n".join(html)


class Bootstrap_Accordion:
    def __init__(self, data=[], name='AccordionExample', args=None):
        self.data = data
        self.name = name
        self.args = args

    def __html_accordion_btn(self, txt="", collapsed=False, target=""):
        attrs = {'type': "button",
                 'data-bs-target': "#%s" % target,
                 'data-bs-toggle': "collapse",
                 'aria-expanded': "true",
                 'aria-controls': "%s" % target,
                 'class': "accordion-button %s" % ("collapsed" if collapsed else ""),
                 }
        return HtmlHelper().cblock("button", attrs=attrs, content=txt)
        # return "<button %s>%s</button>" % (" ".join(
        #     ["%s=\"%s\"" % (key, attrs[key]) for key in attrs.keys() if attrs[key] != ""]), txt)

    def __html_accordion_item(self, title="", body="", itemID="", collapsed=False):
        html = []
        BH = lambda n: HtmlHelper(indent=n)
        html.append(BH(0).block("div", attrs={"class": "accordion-item"}))
        html.append(BH(2).block("h2", attrs={"class": "accordion-header", "id": "%s-heading" % itemID}))
        html.append("    %s" % self.__html_accordion_btn(txt=title, collapsed=collapsed, target=itemID))
        html.append(BH(2).block("/h2"))
        html.append(BH(2).block("div", attrs={"id": "%s" % itemID,
                                              "class": "accordion-collapse collapse %s" % ("show" if not collapsed else ""),
                                              "aria-labelledby": "%s-heading" % itemID}))
        html.append(BH(4).block("div", attrs={"class": "accordion-body"}))
        html.append("      %s" % body)
        html.append(BH(4).block("/div"))
        html.append(BH(2).block("/div"))
        html.append(BH(0).block("/div"))
        return "\n".join(html)

    @property
    def html(self):
        html = []
        html.append(HtmlHelper().block("div", attrs={"class": "accordion w-100", "id": self.name}))
        for ii, item in enumerate(self.data):
            item_html = self.__html_accordion_item(title=item['title'],
                                                   body=item['body'],
                                                   itemID="%s-item%i" % (self.name, ii),
                                                   collapsed=ii != 0)
            html.append(item_html)
        html.append(HtmlHelper().block("/div"))
        return "\n".join(html)


class Bootstrap_Figure:
    def __init__(self, src=None, alt="", caption=""):
        """Return a Boostrap Figure html"""
        self.src = src
        self.alt = alt
        self.caption = caption

    @property
    def html(self):
        html = []
        # html.append("<figure class=\"figure\">")
        # html.append("  <img src=\"{src}\" class=\"figure-img img-fluid rounded\" alt=\"{alt}\">")
        # html.append("  <figcaption class=\"figure-caption\">{caption}</figcaption>")
        # html.append("</figure>")
        html.append(HtmlHelper(indent=0).block("figure", attrs={"class": "figure"}))
        html.append(HtmlHelper(indent=2).block("img", attrs={"class": "figure-img img-fluid rounded",
                                                             "src": "{src}",
                                                             "alt": "{alt}"}))
        html.append(HtmlHelper(indent=2).cblock("figcaption", attrs={"class": "figure-caption"}, content="{caption}"))
        html.append(HtmlHelper(indent=0).block("/figure"))
        html = "\n".join(html).format(src=self.src, alt=self.alt, caption=self.caption)
        return html


def parse_args(wmo, cyc):
    """Return request parameters as an Args instance"""
    WMO = int(escape(wmo))
    CYC = int(escape(cyc))
    args = Args(WMO, CYC, json=True)
    args.nfloats = request.args.get('nfloats', args.__getattribute__('nfloats'), int)
    args.velocity = request.args.get('velocity', args.__getattribute__('velocity'), str)
    return args


@app.route('/', defaults={'wmo': None, 'cyc': None}, methods=['GET', 'POST'])
@app.route('/<int:wmo>/<int:cyc>', methods=['GET', 'POST'])
def index(wmo, cyc):
    # Parse request parameters:
    wmo = wmo if wmo is not None else 0
    cyc = cyc if cyc is not None else 0
    args = parse_args(wmo, cyc)

    if wmo == 0:
        template_data = {'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
                         'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
                         'runs_html': get_html_of_simulations_accordion(args.output, request.base_url),
                         'app_url': request.url_root,
                         'css': url_for("static", filename="css")}
        # print(template_data['runs_html'])

        html = render_template('index2.html', **template_data)
        return html

    else:
        return redirect(url_for('results', wmo=args.wmo, cyc=args.cyc, nfloats=args.nfloats, velocity=args.velocity))


@app.route('/recap', defaults={'wmo': None}, methods=['GET', 'POST'])
@app.route('/recap/<int:wmo>', methods=['GET', 'POST'])
def recap(wmo):
    # Parse request parameters:
    wmo = wmo if wmo is not None else 0
    args = parse_args(wmo, 0)
    nfloats = args.nfloats
    velocity = args.velocity
    figure = request.args.get('figure', 'metrics', str)

    # Get list of figures
    src = os.path.abspath(os.path.sep.join([".", "static"]))
    if figure == 'metrics' or figure == 'metric':
        figure_pattern = "vfrecov_metrics01_%s_%s.png" % (velocity, nfloats)
    elif figure == 'predictions' or figure == 'prediction':
        figure_pattern = "vfrecov_predictions_recap_%s_%s.png" % (velocity, nfloats)
    elif figure == 'details' or figure == 'detail':
        figure_pattern = "vfrecov_predictions_%s_%s.png" % (velocity, nfloats)
    elif figure == 'flow':
        figure_pattern = "vfrecov_velocity_%s.png" % velocity
    # print(nfloats, velocity, figure, figure_pattern)

    if wmo != 0:
        flist = sorted(glob.glob(os.path.sep.join([src, "data", str(wmo), "*", figure_pattern])))
    else:
        flist = sorted(glob.glob(os.path.sep.join([src, "data", "*", "*", figure_pattern])))
    slist = []
    for filename in flist:
        f = filename.replace(src, "")
        url = url_for('static', filename=f)
        url = os.path.normpath(url)
        if url is not None:
            slist.append(url)

    carousel_html = Bootstrap_Carousel(slist, 'recapCarousel', args).html if len(slist) > 0 else None
    template_data = {'css': url_for("static", filename="css"),
                     'js': url_for("static", filename="js"),
                     'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
                     'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
                     'carousel_html': carousel_html,
                     'WMO': args.wmo,
                     'CYC': args.cyc,
                     'VELOCITY': args.velocity,
                     'NFLOATS': args.nfloats,
                     'file_number': len(slist),
                     'app_url': request.url_root,
                     'url_recap': url_for("recap", wmo=args.wmo, nfloats=args.nfloats, velocity=args.velocity),
                     'url_map': url_for("map", wmo=args.wmo, nfloats=args.nfloats, velocity=args.velocity),
                     'WMO': args.wmo if args.wmo > 0 else None,
                     'ea_float': argopy.dashboard(argopy.utilities.check_wmo(args.wmo), url_only=True) if args.wmo > 0 else None,
                     }

    html = render_template('list2.html', **template_data)
    return html


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
        'VELOCITY': args.velocity,
        'NFLOATS': args.nfloats,
        'url': request.base_url,
        'url_predict': url_for("predict", wmo=args.wmo, cyc=args.cyc, nfloats=args.nfloats, velocity=args.velocity),
        'url_recap': url_for("recap", wmo=args.wmo, nfloats=args.nfloats, velocity=args.velocity),
        'url_map': url_for("map", wmo=args.wmo, nfloats=args.nfloats, velocity=args.velocity),
        'prediction_src': None,
        'metric_src': None,
        'velocity_src': None,
        'prediction_recap_src': None,
        'data_js': None,
        'prediction_lon': None,
        'prediction_lat': None,
        'prediction_score': None,
        'error_transit': None,
        'error_bearing': None,
        'error_dist': None,
    }

    # Load data for this set-up:
    jsdata = load_data_for(args)
    # print(jsdata)

    if jsdata is not None:
        template_data['data_js'] = url_for('predict', **args.amap)

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
        if 'score' in jsdata['prediction_metrics']['pairwise_distances']:
            template_data['prediction_score'] = "%0.0f%%" % (100*float(jsdata['prediction_metrics']['pairwise_distances']['score']['value']))
            # template_data['prediction_score'] = "%0.0f%%" % (100*float(jsdata['prediction_metrics']['pairwise_distances']['overlapping']['value']))
        else:
            template_data['prediction_score'] = "?"

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

    template_data['ea_float'] = argopy.dashboard(argopy.utilities.check_wmo(args.wmo), url_only=True)
    template_data['ea_profile'] = argopy.dashboard(argopy.utilities.check_wmo(args.wmo),
                                                   argopy.utilities.check_cyc(args.cyc), url_only=True)

    html = render_template('results2.html', **template_data)
    return html


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


# @app.route('/results/<int:wmo>/<int:cyc>', methods=['GET'])
def results_deprec(wmo, cyc):
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
        template_data['data_js'] = url_for('predict', **args.amap)

        template_data['prediction_src'] = jsdata['meta']['figures']['predictions']
        template_data['prediction_recap_src'] = jsdata['meta']['figures']['predictions_recap']
        template_data['velocity_src'] = jsdata['meta']['figures']['velocity']
        template_data['metric_src'] = jsdata['meta']['figures']['metrics']

        template_data['prediction_lon'] = "%0.3f" % jsdata['prediction_location']['longitude']['value']
        template_data['prediction_lon_unit'] = "%s" % jsdata['prediction_location']['longitude']['unit'].replace("degree", "deg")
        template_data['prediction_lat'] = "%0.3f" % jsdata['prediction_location']['latitude']['value']
        template_data['prediction_lat_unit'] = "%s" % jsdata['prediction_location']['latitude']['unit'].replace("degree", "deg")

        template_data['error_bearing'] = "%0.1f" % jsdata['prediction_location_error']['bearing']['value']
        template_data['error_bearing_unit'] = "%s" % jsdata['prediction_location_error']['bearing']['unit']
        template_data['error_dist'] = "%0.1f" % jsdata['prediction_location_error']['distance']['value']
        template_data['error_dist_unit'] = "%s" % jsdata['prediction_location_error']['distance']['unit']

    template_data['ea_float'] = argopy.dashboard(argopy.utilities.check_wmo(args.wmo), url_only=True)
    template_data['ea_profile'] = argopy.dashboard(argopy.utilities.check_wmo(args.wmo),
                                                   argopy.utilities.check_cyc(args.cyc), url_only=True)

    html = render_template('results1.html', **template_data)
    return html


@app.route('/data_old/<int:wmo>/<int:cyc>', methods=['GET'])
def data_old(wmo, cyc):
    # Parse request parameters:
    args = parse_args(wmo, cyc)
    return redirect(url_for('predict', wmo=args.wmo, cyc=args.cyc, nfloats=args.nfloats, velocity=args.velocity))


@app.route('/data', defaults={'wmo': None}, methods=['GET', 'POST'])
@app.route('/data/<int:wmo>', methods=['GET', 'POST'])
def data(wmo):
    # Parse request parameters:
    wmo = wmo if wmo is not None else 0
    args = parse_args(wmo, 0)

    src = os.path.abspath(os.path.sep.join([".", "static"]))
    filepattern = "prediction_%s_%i.json" % (args.velocity, args.nfloats)
    if wmo != 0:
        flist = sorted(glob.glob(os.path.sep.join([src, "data", str(wmo), "*", filepattern])))
    else:
        flist = sorted(glob.glob(os.path.sep.join([src, "data", "*", "*", filepattern])))
    slist = []
    for filename in flist:
        f = filename.replace(src, "")
        url = url_for('static', filename=f)
        url = os.path.normpath(url)
        if url is not None:
            slist.append(url)

    feature_list = []
    for filename in flist:
        this_wmo = filename.split(os.path.sep)[-3]
        this_cyc = filename.split(os.path.sep)[-2]
        jsdata = load_data_for(Args(this_wmo, this_cyc, nfloats=args.nfloats, velocity=args.velocity))
        f = Feature(geometry=Point(
            (jsdata['prediction_location']['longitude']['value'], jsdata['prediction_location']['latitude']['value'])),
                    properties=jsdata)
        feature_list.append(f)

    return jsonify(FeatureCollection(feature_list))


@app.route('/map', defaults={'wmo': None}, methods=['GET', 'POST'])
@app.route('/map/<int:wmo>', methods=['GET', 'POST'])
def map(wmo):
    # Parse request parameters:
    wmo = wmo if wmo is not None else 0
    args = parse_args(wmo, 0)

    template_data = {'css': url_for("static", filename="css"),
                     'js': url_for("static", filename="js"),
                     'dist': url_for("static", filename="dist"),
                     'cdn_bootstrap': 'cdn.jsdelivr.net/npm/bootstrap@5.2.2',
                     'cdn_prism': 'cdn.jsdelivr.net/npm/prismjs@1.29.0',
                     'app_url': request.url_root,
                     'url_recap': url_for("recap", wmo=args.wmo, nfloats=args.nfloats, velocity=args.velocity),
                     'url_map': url_for("map", wmo=args.wmo, nfloats=args.nfloats, velocity=args.velocity),
                     'WMO': args.wmo if args.wmo != 0 else None,
                     'CYC': args.cyc if args.wmo != 0 else None,
                     'VELOCITY': args.velocity,
                     'NFLOATS': args.nfloats,
                     'jsdata': url_for('data', wmo=args.wmo if args.wmo != 0 else None,
                                       nfloats=args.nfloats, velocity=args.velocity)
                     }
    # print(jsonify(template_data))

    html = render_template('map.html', **template_data)
    return html

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
