import os
import glob
import numpy as np
from flask import request, url_for
from .for_flask import parse_args, request_opts_for_data, read_params_from_path
from dominate.tags import button, div, h2, h5, p, a, span
from dominate.tags import img, figure, figcaption
from dominate.util import raw


def get_html_of_simulations_list_deprecated(this_src, this_urlroot):
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


def get_html_of_simulations_accordion_deprecated(this_src, this_urlroot):
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


class Bootstrap_Figure:

    def __init__(self, src=None, alt="", caption=""):
        """Return a Boostrap Figure html

        >>> Bootstrap_Figure(src='logo-virtual-fleet-recovery.png', caption='coucou').html
        """
        self.src = src
        self.alt = alt
        self.caption = caption

    @property
    def html(self):
        f = figure(cls="figure")
        with f:
            img(src=self.src, alt=self.alt, cls="figure-img img-fluid rounded")
            figcaption(self.caption, cls='figure-caption')
        return f


class Bootstrap_Accordion:

    def __init__(self, data=[], name='AccordionExample', args=None):
        self.data = data
        self.name = name
        self.args = args

    def _html_accordion_btn(self, txt="", collapsed=False, target=""):
        b = button(txt,
                   type='button',
                   data_bs_target="#%s" % target,
                   data_bs_toggle="collapse",
                   aria_expanded="true",
                   aria_controls="%s" % target,
                   cls="accordion-button %s" % ("collapsed" if collapsed else ""))
        return b

    def _html_accordion_item(self, title="", body="", itemID="", collapsed=False):
        d = div(cls="accordion-item")
        d += h2(self._html_accordion_btn(txt=title, collapsed=collapsed, target=itemID), cls="accordion-header",
                id="%s-heading" % itemID)
        d += div(id="%s" % itemID, cls="accordion-collapse collapse %s" % ("show" if not collapsed else ""),
                 aria_labelledby="%s-heading" % itemID)
        d += div(raw(body), cls="accordion-body")
        return d

    @property
    def html(self):
        code = div(cls="accordion w-100", id=self.name)
        for ii, item in enumerate(self.data):
            with code:
                item_html = self._html_accordion_item(title=item['title'],
                                                      body=item['body'],
                                                      itemID="%s-item%i" % (self.name, ii),
                                                      collapsed=ii != 0)
                item_html
        return code


class Bootstrap_Carousel:
    """Return a Boostrap carousel, a slideshow component for cycling through elements—images or slides of text—like a carousel.

    https://getbootstrap.com/docs/4.0/components/carousel/

    Pattern:
    ```html
        <div id="carouselExampleIndicators" class="carousel slide">

          <div class="carousel-indicators">
            <button type="button" data-bs-target="#carouselExampleIndicators" data-bs-slide-to="0" class="active" aria-current="true" aria-label="Slide 1"></button>
            <button type="button" data-bs-target="#carouselExampleIndicators" data-bs-slide-to="1" aria-label="Slide 2"></button>
            <button type="button" data-bs-target="#carouselExampleIndicators" data-bs-slide-to="2" aria-label="Slide 3"></button>
          </div>

          <div class="carousel-inner">
            <div class="carousel-item active">
              <img src="..." class="d-block w-100" alt="...">
              <div class="carousel-caption d-none d-md-block">
                <h5>First slide label</h5>
                <p>Some representative placeholder content for the first slide.</p>
              </div>
            </div>
            <div class="carousel-item">
              <img src="..." class="d-block w-100" alt="...">
            </div>
            <div class="carousel-item">
              <img src="..." class="d-block w-100" alt="...">
            </div>
          </div>


          <button class="carousel-control-prev" type="button" data-bs-target="#carouselExampleIndicators" data-bs-slide="prev">
            <span class="carousel-control-prev-icon" aria-hidden="true"></span>
            <span class="visually-hidden">Previous</span>
          </button>
          <button class="carousel-control-next" type="button" data-bs-target="#carouselExampleIndicators" data-bs-slide="next">
            <span class="carousel-control-next-icon" aria-hidden="true"></span>
            <span class="visually-hidden">Next</span>
          </button>

        </div>
    ```
    """

    def __init__(self, figure_list=[], id='carouselExample', args=None, opts=None):
        """Create a Bootstrap Carousel for a given list of figure files"""
        self.flist = figure_list
        self.id = id
        self.args = args
        self.opts = opts

    def __repr__(self):
        summary = []
        summary.append("<bootstrap.carouselWithCaption>")
        summary.append("Figures: %i" % len(self.flist))
        summary.append("ID: %s" % self.id)
        return "\n".join(summary)

    def _html_carousel_indicators_btn(self, islide=0, active=False, target: str = None):
        b = button(type='button',
                   data_bs_target="#%s" % self.id if target is None else target,
                   data_bs_slide_to="%i" % islide,
                   aria_label="Slide %i" % int(islide + 1),
                   aria_current="true" if active else "false",
                   cls="active" if active else "")
        return b

    def get_list_of_carousel_indicators_btn(self):
        d = div(cls="carousel-indicators")
        for islide in np.arange(0, len(self.flist)):
            d += self._html_carousel_indicators_btn(islide, active=islide == 0, target=self.id)
        return d

    def _get_results_lnk(self, wmo, cyc, params):
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

    def _get_recap_lnk(self, wmo, cyc):
        opts = request_opts_for_data(request, parse_args(wmo, cyc))
        opts.pop('cyc')
        results_lnk = url_for('.recap', **opts)
        return results_lnk

    def _get_carousel_item(self,
                            src='...',
                            label='Slide label',
                            description='Some representative placeholder content for the second slide.',
                            active=False):
        d = div(cls="carousel-item %s" % ("active" if active else ""), data_bs_interval=10)
        d += img(src=src, cls='d-block w-100', alt='')
        d += div([h5(label), p(raw(description))], cls="carousel-caption d-none d-md-block")
        return d

    def get_list_of_carousel_items(self):
        d = div(cls="carousel-inner")
        for islide, figure_file in enumerate(self.flist):

            # Get data to make description links:
            params = read_params_from_path(figure_file, plist=['VEL', 'NF', 'CYCDUR', 'PDPTH'])
            wmo = figure_file.replace("static/data/", "").split("/")[1]
            cyc = figure_file.replace("static/data/", "").split("/")[2]

            results_lnk = self._get_results_lnk(wmo, cyc, params)
            recap_lnk = self._get_recap_lnk(wmo, cyc)
            description = "%s / %s" % (a("Swipe only this float", href=recap_lnk, target=""),
                                       a("Check this cycle details", href=results_lnk, target=""))

            label = "Float %s - Cycle %s" % (wmo, cyc)
            d += self._get_carousel_item(src=figure_file, label=label, description=description, active=islide == 0)
        return d

    def get_carousel_controls(self):
        b1 = button(
            [span(cls="carousel-control-prev-icon", aria_hidden="true"), span("Previous", cls="visually-hidden")],
            type='button',
            data_bs_target="#%s" % self.id,
            data_bs_slide="prev",
            cls="carousel-control-prev")
        b2 = button([span(cls="carousel-control-next-icon", aria_hidden="true"), span("Next", cls="visually-hidden")],
                    type='button',
                    data_bs_target="#%s" % self.id,
                    data_bs_slide="next",
                    cls="carousel-control-next")
        return (b1, b2)

    @property
    def html(self):
        d = div(id=self.id, cls="carousel carousel-dark slide", data_bs_ride="false")

        d += self.get_list_of_carousel_indicators_btn()
        d += self.get_list_of_carousel_items()

        # Add control buttons:
        for b in self.get_carousel_controls():
            d += b

        return d