import numpy as np
import os
import json
import glob
from flask import request, url_for
from markupsafe import escape
from myapp import app


APP_NAME = __name__.split('.')[0]
# print("myapp/utils/for_flask.py:", app.config)


class Args:

    def __init__(self, wmo, cyc, default=True, *args, **kwargs):
        self.wmo = wmo
        self.cyc = cyc
        self.vf = None
        if 'nfloats' in kwargs:
            self.nfloats = kwargs['nfloats']
        elif default:
            self.nfloats = app.config['DEFAULT_PARAMS']['nfloats']
        else:
            self.nfloats = None

        if 'velocity' in kwargs:
            self.velocity = kwargs['velocity']
        elif default:
            self.velocity = app.config['DEFAULT_PARAMS']['velocity']
        else:
            self.velocity = None

        if 'output' in kwargs:
            self.output = kwargs['output']
        else:
            self.output = os.path.join(app.config['DATASTORE'], 'data')

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

        if 'cfg_parking_depth' in kwargs:
            self.cfg_parking_depth = kwargs['cfg_parking_depth']
        elif default:
            self.cfg_parking_depth = app.config['DEFAULT_PARAMS']['cfg_parking_depth']
        else:
            self.cfg_parking_depth = None

        if 'cfg_cycle_duration' in kwargs:
            self.cfg_cycle_duration = kwargs['cfg_cycle_duration']
        elif default:
            self.cfg_cycle_duration = app.config['DEFAULT_PARAMS']['cfg_cycle_duration']
        else:
            self.cfg_cycle_duration = None

    def __iter__(self):
        self.__i = 0
        self.__l = ['wmo', 'cyc', 'nfloats', 'velocity', 'cfg_parking_depth', 'cfg_cycle_duration']
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
        m = {'wmo': self.wmo,
             'cyc': self.cyc,
             'nfloats': self.nfloats,
             'velocity': self.velocity,
             'cfg_parking_depth': self.cfg_parking_depth,
             'cfg_cycle_duration': self.cfg_cycle_duration}
        return m

    def html(self):
        summary = [""]
        summary.append("WMO: %i" % self.wmo)
        summary.append("CYC: %i" % self.cyc)
        summary.append("nfloats: %i" % self.nfloats)
        summary.append("velocity: %s" % self.velocity)
        summary.append("cfg_parking_depth [db]: %i" % self.cfg_parking_depth)
        summary.append("cfg_cycle_duration [hours]: %s" % self.cfg_cycle_duration)
        # summary.append("Simulation dashboard: %s" % "" )
        summary.append("<hr>")
        summary.append("<b>VirtualFleet Recovery</b>")
        summary.append("(c) Argo-France/Ifremer/LOPS, 2022")
        return "<br>".join(summary)


def parse_args(wmo, cyc, default=True):
    """Return request parameters as an Args instance"""
    WMO = int(escape(wmo)) if wmo is not None else None
    CYC = int(escape(cyc)) if cyc is not None else None
    args = Args(WMO, CYC, default=default, json=True)
    args.nfloats = request.args.get('nfloats', args.__getattribute__('nfloats'), int)
    args.velocity = request.args.get('velocity', args.__getattribute__('velocity'), str)
    args.cfg_parking_depth = request.args.get('cfg_parking_depth', args.__getattribute__('cfg_parking_depth'), int)
    args.cfg_cycle_duration = request.args.get('cfg_cycle_duration', args.__getattribute__('cfg_cycle_duration'), int)
    return args


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
    url = "/".join([request.base_url, url_for('.static', filename=this_path)])

    if 'predict/' in url:
        url = url.replace("predict/%i/%i//" % (this_args.wmo, this_args.cyc), "")
    elif 'results/' in url:
        # print("results/%i/%i//" % (this_args.wmo, this_args.cyc))
        url = url.replace("results/%i/%i//" % (this_args.wmo, this_args.cyc), "")
    elif 'trigger/' in url:
        url = url.replace("trigger/%i/%i//" % (this_args.wmo, this_args.cyc), "")
    elif 'data/' in url:
        url = url.replace("data//static/data/", "static/data/")
        # /data/6901925//static/data/6901925 > /static/data/6901925
        url = url.replace("/data/%s//static/data/" % this_args.wmo, "/static/data/")
        # url = url.replace("//static/data/", "")
    # url = url.replace("//", "/")
    # print(filename, "\n", url)
    if safe:
        local_file = os.path.abspath(os.path.sep.join([".", APP_NAME, "static", "data",
                                                       str(this_args.wmo), str(this_args.cyc),
                                                       filename]))
        # local_file = os.path.sep.join([this_args.output, , filename])
        # local_file = os.path.abspath(local_file)
        if os.path.lexists(local_file):
            # print(local_file)
            return url
        else:
            print("%s not found" % local_file)
            return None
    return url


def get_sim_suffix(this_args, legacy=False):
    """Compose a string suffix for output files"""
    # Must return similar output than the cli version
    if legacy:
        suf = '%s_%i' % (this_args.velocity, this_args.nfloats)
    else:
        # suf = 'VEL%s_NF%i_CYCDUR%i_PDPTH%i' % (this_args.velocity,
        #                                        this_args.nfloats,
        #                                        int(this_args.cfg_cycle_duration),
        #                                        int(this_args.cfg_parking_depth))
        # suf = 'VEL%s_NF%i' % (this_args.velocity, this_args.nfloats)
        if this_args.velocity is not None:
            suf = "VEL%s" % this_args.velocity
        else:
            suf = "VEL*"
        if this_args.nfloats is not None:
            suf += "_NF%i" % this_args.nfloats
        else:
            suf += "_NF*"
        if this_args.cfg_cycle_duration is not None:
            suf += "_CYCDUR%i" % this_args.cfg_cycle_duration
        else:
            suf += "_CYCDUR*"
        if this_args.cfg_parking_depth is not None:
            suf += "_PDPTH%i" % this_args.cfg_parking_depth
        else:
            suf += "_PDPTH*"

    return suf


def get_sim_files(this_args, legacy=False):
    js = os.path.sep.join([
                           "data",
                           str(this_args.wmo),
                           str(this_args.cyc),
                           "prediction_%s.json" % get_sim_suffix(this_args, legacy=legacy)])
    ajs = os.path.abspath(os.path.sep.join([".", APP_NAME, "static", js]))
    ajs_list = sorted(glob.glob(ajs))
    return ajs_list


def complete_data_for(this_args, this_js, legacy=False):
    """Return API complemented json data for a simulation

    Simulation parameters are determined using args
    """
    suffix = get_sim_suffix(this_args, legacy=legacy)
    if "*" in suffix:
        raise ValueError("We can complement json data structure only for a single simulation, a wild card was found")
    figlist = {'predictions': simulation_file_url(this_args,
                                                  "vfrecov_predictions_%s.png" % suffix,
                                                  safe=True),
               'metrics': simulation_file_url(this_args,
                                              "vfrecov_metrics01_%s.png" % suffix,
                                              safe=True),
               'velocity': simulation_file_url(this_args,
                                               "vfrecov_velocity_%s.png" % this_args.velocity),
               'positions': simulation_file_url(this_args,
                                                "vfrecov_positions_%s.png" % suffix,
                                                safe=True),
               'predictions_recap': simulation_file_url(this_args,
                                                        "vfrecov_predictions_recap_%s.png" % suffix,
                                                        safe=True)}
    this_js['meta']['figures'] = figlist

    this_js['meta']['api'] = {'cycle_page': "".join([request.host_url[0:-1],
                                                     url_for(".index",
                                                             wmo=this_args.wmo,
                                                             cyc=this_args.cyc,
                                                             nfloats=this_args.nfloats,
                                                             velocity=this_args.velocity)]),
                              'float_page': "".join([request.host_url[0:-1],
                                                     url_for(".recap",
                                                             wmo=this_args.wmo,
                                                             nfloats=this_args.nfloats,
                                                             velocity=this_args.velocity)]),
                              'float_map': "".join([request.host_url[0:-1],
                                                    url_for(".map_error",
                                                            wmo=this_args.wmo,
                                                            nfloats=this_args.nfloats,
                                                            velocity=this_args.velocity)]),
                              }

    return this_js


def load_data_for(this_args, legacy=False):
    """Return the complete json file data from a single simulation

    Simulation parameters are determined using `args`

    If more than one simulation correspond the `args` parameters, return data for the 1st simulation

    Raw data are complemented with results from complete_data_for() function
    """
    ajs_list = get_sim_files(this_args, legacy=legacy)
    if (len(ajs_list) > 0) and os.path.exists(ajs_list[0]):
        # Work with the 1st available simulation set of files:
        ajs = ajs_list[0]
        # print(ajs)
        with open(ajs) as f:
            jsdata = json.load(f)
        # Update `args` with the selected simulation
        this_args.nfloats = int(jsdata['meta']['Nfloats'])
        this_args.velocity = jsdata['meta']['Velocity field']
        if 'VFloats_config' in jsdata['meta']:
            this_args.cfg_cycle_duration = int(jsdata['meta']['VFloats_config']['data']['cycle_duration']['value'])
            this_args.cfg_parking_depth = int(jsdata['meta']['VFloats_config']['data']['parking_depth']['value'])
        else:
            this_args.cfg_cycle_duration = 240
            this_args.cfg_parking_depth = 1000
        # print(this_args.amap)
        # print(jsdata)
    else:
        print('No data found at: ', ajs_list)
        jsdata = None

    if jsdata is not None:
        jsdata = complete_data_for(this_args, jsdata, legacy=legacy)

    return jsdata, this_args


def request_opts_for_data(req, this_args):
    opts = {}
    if isinstance(req, dict):
        request_args = req['args']
    else:
        request_args = req.args

    opts['wmo'] = this_args.wmo if this_args.wmo != 0 else None
    opts['cyc'] = this_args.cyc if this_args.cyc != 0 else None

    if 'velocity' in request_args:
        opts['velocity'] = this_args.velocity
    else:
        opts['velocity'] = None

    if 'nfloats' in request_args:
        opts['nfloats'] = int(this_args.nfloats)
    else:
        opts['nfloats'] = None

    if 'cfg_cycle_duration' in request_args:
        opts['cfg_cycle_duration'] = int(this_args.cfg_cycle_duration)
    else:
        opts['cfg_cycle_duration'] = None

    if 'cfg_parking_depth' in request_args:
        opts['cfg_parking_depth'] = int(this_args.cfg_parking_depth)
    else:
        opts['cfg_parking_depth'] = None

    # print(req, this_args, opts)
    return opts


def read_params_from_path(pathname, plist=['VEL', 'NF', 'CYCDUR', 'PDPTH']):
    filename = os.path.splitext(os.path.split(pathname)[-1])[0]
    startwith = lambda s, w: w == s[0:len(w)] if len(s) > len(w) else False
    result = {}
    for part in filename.split("_"):
        for param in plist:
            if startwith(part, param):
                result[param] = part.replace(param, '')
    return result


def search_local_prediction_datafiles(this_args):

    wmo = this_args.wmo

    # src = os.path.abspath(os.path.sep.join([".", "static"]))
    src = app.config["DATASTORE"]

    def get_a_search_pattern(req, this_args):
        # Should be coherent with get_sim_suffix:
        filepattern = "prediction"
        # print('request.args', req.args)

        if 'velocity' in req.args:
            filepattern += "_VEL%s" % this_args.velocity
        else:
            filepattern += "_VEL*"

        if 'nfloats' in req.args:
            filepattern += "_NF%i" % int(this_args.nfloats)
        else:
            filepattern += "_NF*"

        if 'cfg_cycle_duration' in req.args:
            filepattern += "_CYCDUR%i" % int(this_args.cfg_cycle_duration)
        else:
            filepattern += "_CYCDUR*"

        if 'cfg_parking_depth' in req.args:
            filepattern += "_PDPTH%i" % int(this_args.cfg_parking_depth)
        else:
            filepattern += "_PDPTH*"

        filepattern += ".json"
        # print(filepattern)
        return filepattern

    filepattern = get_a_search_pattern(request, this_args)
    # print(filepattern, src, this_args.amap)

    if wmo != 0:
        flist = sorted(glob.glob(os.path.sep.join([src, "data", str(wmo), "*", filepattern])))
    else:
        flist = sorted(glob.glob(os.path.sep.join([src, "data", "*", "*", filepattern])))
    # print(filepattern, flist)

    slist = []
    for filename in flist:
        f = filename.replace(src, "")
        url = url_for('static', filename=f)
        url = os.path.normpath(url)
        if url is not None:
            slist.append(url)
    # print(filepattern, len(slist))

    return slist


def search_local_prediction_figfiles(this_args, this_request):
    figure_name = this_request.args.get('figure', 'metrics', str)

    def get_a_search_pattern_suffix(this_args, this_request):
        # Should be coherent with get_sim_suffix:

        if 'velocity' in this_request.args:
            filepattern = "VEL%s" % this_args.velocity
        else:
            filepattern = "VEL*"

        if 'nfloats' in this_request.args:
            filepattern += "_NF%i" % int(this_args.nfloats)
        else:
            filepattern += "_NF*"

        if 'cfg_cycle_duration' in this_request.args:
            filepattern += "_CYCDUR%i" % int(this_args.cfg_cycle_duration)
        else:
            filepattern += "_CYCDUR*"

        if 'cfg_parking_depth' in this_request.args:
            filepattern += "_PDPTH%i" % int(this_args.cfg_parking_depth)
        else:
            filepattern += "_PDPTH*"

        filepattern += ".png"
        # print(filepattern)
        return filepattern

    # Get list of figures
    # src = os.path.abspath(os.path.sep.join([".", "static"]))
    src = app.config["DATASTORE"]

    figure_suffix = get_a_search_pattern_suffix(this_args, this_request)
    if figure_name == 'metrics' or figure_name == 'metric':
        figure_pattern = "vfrecov_metrics01_%s" % figure_suffix
    elif figure_name == 'predictions' or figure_name == 'prediction':
        figure_pattern = "vfrecov_predictions_recap_%s" % figure_suffix
    elif figure_name == 'details' or figure_name == 'detail':
        figure_pattern = "vfrecov_predictions_%s" % figure_suffix
    elif figure_name == 'flow':
        figure_pattern = "vfrecov_velocity_%s.png" % this_args.velocity

    if this_args.wmo is not None:
        pattern = os.path.sep.join([src, "data", str(this_args.wmo), "*", figure_pattern])
    else:
        pattern = os.path.sep.join([src, "data", "*", "*", figure_pattern])
    flist = sorted(glob.glob(pattern))
    slist = []
    for filename in flist:
        f = filename.replace(src, "")
        url = url_for('static', filename=f)
        url = os.path.normpath(url)
        if url is not None:
            slist.append(url)

    # print(this_args.amap, figure_name, figure_suffix, pattern, len(slist))

    return slist
