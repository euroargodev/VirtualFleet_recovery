"""
Module for HTML helpers and custom Bootstrap components
"""

from flask import request, url_for
from .for_flask import parse_args, request_opts_for_data, read_params_from_path
from dominate.tags import a
from .bootstrap import Bootstrap_Carousel


class Bootstrap_Carousel_Recovery(Bootstrap_Carousel):

    def read_data(self, figure_file):
        params = read_params_from_path(figure_file, plist=['VEL', 'NF', 'CYCDUR', 'PDPTH'])
        wmo = figure_file.replace("static/data/", "").split("/")[1]
        cyc = figure_file.replace("static/data/", "").split("/")[2]
        return wmo, cyc, params

    def results_lnk(self, wmo, cyc, params):
        """Return link to an individual cycle result page"""
        this_figure_args = parse_args(wmo, cyc)
        this_figure_request = {'args': {}}
        for p in params:
            if p == 'VEL':
                this_figure_request['args']['velocity'] = params[p]
                this_figure_args.velocity = params[p]
            if p == 'NF':
                this_figure_request['args']['nfloats'] = params[p]
                this_figure_args.nfloats = params[p]
            if p == 'CYCDUR':
                this_figure_request['args']['cfg_cycle_duration'] = params[p]
                this_figure_args.cfg_cycle_duration = params[p]
            if p == 'PDPTH':
                this_figure_request['args']['cfg_parking_depth'] = params[p]
                this_figure_args.cfg_parking_depth = params[p]
        results_lnk = url_for('.results', **request_opts_for_data(this_figure_request, this_figure_args))
        return results_lnk

    def recap_lnk(self, wmo, cyc):
        """Return link to a float recap page"""
        opts = request_opts_for_data(request, parse_args(wmo, cyc))
        opts.pop('cyc')
        results_lnk = url_for('.recap', **opts)
        return results_lnk

    def _get_label(self, islide, figure_file):
        wmo, cyc, params = self.read_data(figure_file)
        return "Float %s - Cycle %s" % (wmo, cyc)

    def _get_description(self, islide, figure_file):
        wmo, cyc, params = self.read_data(figure_file)
        results_lnk = self.results_lnk(wmo, cyc, params)
        recap_lnk = self.recap_lnk(wmo, cyc)
        description = "%s / %s" % (a("Swipe only this float", href=recap_lnk, target=""),
                                   a("Check this cycle details", href=results_lnk, target=""))
        return description
