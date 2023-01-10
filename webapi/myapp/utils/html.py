"""
Module for HTML helpers and custom Bootstrap components
"""

from flask import request, url_for
from .for_flask import parse_args, request_opts_for_data, read_params_from_path
from dominate.tags import a, span, h6, nav, div, ul, li
from dominate.util import raw, text
from .bootstrap import Bootstrap_Carousel
from abc import ABC
import argopy


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


class MenuGroup():
    def __init__(self, header=None, items=None):
        self.header = header
        self.items = []
        if items is not None:
            for item in items:
                self.add_item(item)

    def add_item(self, item):
        if 'txt' in item and 'icon' in item and 'href' in item:
            if 'active' not in item:
                item['active'] = False
            if 'disabled' not in item:
                item['disabled'] = False
            self.items.append(item)
        else:
            raise ValueError("Item is missing a property to be added ('txt', 'icon' or 'href')")

    def activate(self, txt):
        for item in self.items:
            if item['txt'] == txt:
                item['active'] = True
            else:
                item['active'] = False
        return self

    @property
    def serialised(self):
        return {'header': self.header, 'items': self.items}


class Sidebar(ABC):
    """Return a sidebar html

    HTML structure:
    ```html
    <nav id="sidebarMenu" class="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse">

        <div class="position-sticky pt-3 sidebar-sticky">

          <ul class="nav flex-column">
            <li class="nav-item">
              <a class="nav-link" href="#">
                <span data-feather="home" class="align-text-bottom"></span>
                Item 1
              </a>
            </li>
            <li class="nav-item">
              <a class="nav-link active" href="#">
                <span data-feather="github" class="align-text-bottom"></span>
                Item 2
              </a>
            </li>
          </ul>

          <h6 class="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted text-uppercase">
            <span>Menu header</span>
          </h6>

          <ul class="nav flex-column">
            <li class="nav-item">
              <a class="nav-link" href="#">
                <span data-feather="home" class="align-text-bottom"></span>
                Item 3
              </a>
            </li>
            <li class="nav-item">
              <a class="nav-link" href="#">
                <span data-feather="github" class="align-text-bottom"></span>
                Item 4
              </a>
            </li>
          </ul>

        </div>

    </nav>
    ```
    """

    def __init__(self, grps):
        self.grps = grps

    def an_icon(self, name='home'):
        return span(data_feather=name, cls="align-text-bottom")

    def navitem(self, txt="Nav item", href="#", icon='home', active=False, disabled=False):
        html = a(href=href, cls="nav-link %s %s" % ("active" if active else "", "disabled" if disabled else ""))
        with html:
            self.an_icon(icon)
            text(txt)
        return html

    def navheader(self, title="Header"):
        html = h6(
            cls="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted text-uppercase")
        with html:
            span(title)
        return html

    @property
    def html(self):
        n = nav(id="sidebarMenu", cls="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse")
        d = div(cls="position-sticky pt-3 sidebar-sticky")
        with d:
            for menu_groupe in self.grps:
                if menu_groupe.header is not None:
                    d += self.navheader(menu_groupe.header)
                lst = ul(cls="nav flex-column")
                for item in menu_groupe.items:
                    lst += li(self.navitem(txt=item['txt'], href=item['href'], icon=item['icon'], active=item['active'],
                                           disabled=item['disabled']),
                              cls="nav-item")
                d += lst
            # raw("{% include ""footer.html"" %}")
        n += d
        return n.render()


def get_the_sidebar(args, opts, data_js, active=None):
    # m.add_item({'txt':"", 'icon':"", 'href':""})

    grps = []

    m = MenuGroup()
    m.add_item({'txt': "VirtualFleet Recovery", 'icon': "home", 'href': "/"})
    m.add_item({'txt': "Repository", 'icon': "github", 'href': "https://github.com/euroargodev/VirtualFleet_recovery"})
    m.add_item({'txt': "Prediction form", 'icon': "cpu", 'href': url_for(".trigger", **args.amap)})
    if active:
        m.activate(active)
    grps.append(m)

    m = MenuGroup(header='Prediction results')
    m.add_item(
        {'txt': "Download", 'icon': "download", 'href': url_for(".data", **args.amap), 'disabled': data_js == None})
    # m.add_item({'txt':"Synthesis", 'icon':"star", 'href':"#synthesis", 'disabled': data_js == None})
    # m.add_item({'txt':"Figures", 'icon':"image", 'href':"#figures", 'disabled': data_js == None})
    m.add_item({'txt': "Swipe all cycles", 'icon': "list", 'href': url_for(".recap", **opts)})
    m.add_item({'txt': "See on a map", 'icon': "map", 'href': url_for(".map_error", **opts)})
    if active:
        m.activate(active)
    grps.append(m)

    m = MenuGroup(header='Parameters')
    m.add_item({'txt': "Velocity: %s" % (args.velocity if args.velocity else '-'),
                'icon': 'mouse-pointer', 'href': '#', 'disabled': True})
    m.add_item({'txt': "N floats: %s" % (args.nfloats if args.nfloats else '-'),
                'icon': 'life-buoy', 'href': '#', 'disabled': True})
    m.add_item({'txt': "Parking Depth: %s" % (args.cfg_parking_depth if args.cfg_parking_depth else '-'),
                'icon': 'activity', 'href': '#', 'disabled': True})
    m.add_item({'txt': "Cycle Duration: %s" % (args.cfg_cycle_duration if args.cfg_cycle_duration else '-'),
                'icon': 'clock', 'href': '#', 'disabled': True})
    if active:
        m.activate(active)
    grps.append(m)

    m = MenuGroup(header='More')
    if args.wmo and args.wmo != 0:
        m.add_item({'txt': "Float dashboard", 'icon': 'table',
                    'href': argopy.dashboard(argopy.utilities.check_wmo(args.wmo), url_only=True)})
    else:
        m.add_item({'txt': "Float dashboard", 'icon': 'table', 'href': "#", 'disabled': True})
    if data_js is not None and data_js['prediction_location_error']['bearing']['value'] is not None:
        m.add_item({'txt': "Profile page", 'icon': 'anchor', 'href': data_js['profile_to_predict']['url_profile']})
    else:
        m.add_item({'txt': "Profile page", 'icon': 'anchor', 'href': '#', 'disabled': True})
    m.add_item({'txt': "Float Recovery @ Euro-Argo", 'icon': 'zap', 'href': 'https://floatrecovery.euro-argo.eu'})
    m.add_item({'txt': "Recovery Forum", 'icon': 'zap', 'href': 'https://github.com/euroargodev/recovery/issues'})
    if active:
        m.activate(active)
    grps.append(m)

    return Sidebar(grps).html