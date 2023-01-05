import os
import glob
import numpy as np
from flask import request, url_for
from .flask import parse_args, request_opts_for_data, read_params_from_path


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


from dominate.tags import button, div, h2, h5, p, a, span
from dominate.tags import img, figure, figcaption
from dominate.util import raw


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

    def __get_results_lnk(self, wmo, cyc, params):
        this_args = parse_args(wmo, cyc)
        this_request = {'args': {}}
        for p in params:
            if p == 'VEL':
                this_request['args']['velocity'] = params[p]
                this_args.velocity = params[p]
            if p == 'NF':
                this_request['args']['nfloats'] = params[p]
                this_args.nfloats = params[p]
            if p == 'CYCDUR':
                this_request['args']['cfg_cycle_duration'] = params[p]
                this_args.cfg_cycle_duration = params[p]
            if p == 'PDPTH':
                this_request['args']['cfg_parking_depth'] = params[p]
                this_args.cfg_parking_depth = params[p]
        results_lnk = url_for('.results', **request_opts_for_data(this_request, this_args))
        return results_lnk

    def __get_recap_lnk(self, wmo, cyc):
        opts = request_opts_for_data(request, parse_args(wmo, cyc))
        opts.pop('cyc')
        results_lnk = url_for('.recap', **opts)
        return results_lnk

    def __html_carousel_btn(self, islide=0, active=False, target='carouselExample'):
        b = button(type='button',
                   data_bs_target="#%s" % target,
                   data_bs_slide_to="%i" % islide,
                   aria_label="Slide %i" % int(islide + 1),
                   cls="active" if active else "")
        return b

    def __get_list_of_carousel_btn_html(self, this_flist, carouselName='carouselExample'):
        html = []
        for islide in np.arange(0, len(this_flist)):
            html.append(raw(self.__html_carousel_btn(islide, active=islide == 0, target=carouselName)))
        html = "\n".join(html)
        return html

    def __html_carousel_item(self,
                             src='...',
                             label='Slide label',
                             description='Some representative placeholder content for the second slide.',
                             active=False):
        d = div(cls="carousel-item %s" % ("active" if active else ""), data_bs_interval=10)
        d += img(src=src, cls='d-block w-100', alt='')
        d += div([h5(label), p(description)], cls="carousel-caption d-none d-md-block")
        return d

    def __get_list_of_carousel_items_html(self, this_flist):
        html = []
        for islide, figure_file in enumerate(this_flist):
            params = read_params_from_path(figure_file, plist=['VEL', 'NF', 'CYCDUR', 'PDPTH'])
            print(figure_file, params)
            wmo = figure_file.replace("static/data/", "").split("/")[1]
            cyc = figure_file.replace("static/data/", "").split("/")[2]
            label = "Float %s - Cycle %s" % (wmo, cyc)
            results_lnk = self.__get_results_lnk(wmo, cyc, params)
            recap_lnk = self.__get_recap_lnk(wmo, cyc)
            description = "%s / %s" % (raw(a("Swipe only this float", href=recap_lnk, target="")),
                                       raw(a("Check this cycle details", href=results_lnk, target="")))
            html.append(
                self.__html_carousel_item(src=figure_file, label=label, description=description, active=islide == 0))
        html = "\n".join(html)
        return html

    @property
    def html(self):
        d = div(id=self.name, cls="carousel carousel-dark slide", data_bs_ride="false")
        d += div(raw(self.__get_list_of_carousel_btn_html(self.flist, carouselName=self.name)),
                 cls="carousel-indicators")
        d += div(raw(self.__get_list_of_carousel_items_html(self.flist)), cls="carousel-inner")
        d += button(
            [span(cls="carousel-control-prev-icon", aria_hidden="true"), span("Previous", cls="visually-hidden")],
            type='button',
            data_bs_target="#%s" % self.name,
            data_bs_slide="prev",
            cls="carousel-control-prev")
        d += button([span(cls="carousel-control-next-icon", aria_hidden="true"), span("Next", cls="visually-hidden")],
                    type='button',
                    data_bs_target="#%s" % self.name,
                    data_bs_slide="next",
                    cls="carousel-control-next")
        return d


# class HtmlHelper:
#     def __init__(self, indent=0):
#         """HTML string formatting helper
#
#         >>> HtmlHelper().cblock("p", content="Hello !", attrs={"class": "toto", "aria-hidden": "false"})
#         '<p class="toto" aria-hidden="false">Hello !</p>'
#
#         >>> HtmlHelper().block("img", attrs={"src": "fig.png"})
#         '<img src="fig.png">'
#
#         """
#         self.indent = indent
#
#     def __indent(self, txt):
#         shift = " " * self.indent
#         return "%s%s" % (shift, txt)
#
#     def cblock(self, name, attrs={}, content=''):
#         if len(attrs) > 0:
#             html = "<%s %s>%s</%s>" % (
#             name, " ".join(["%s=\"%s\"" % (key, attrs[key]) for key in attrs.keys() if attrs[key] != ""]), content,
#             name)
#         else:
#             html = "<%s>%s</%s>" % (name, content, name)
#         return self.__indent(html)
#
#     def block(self, name, attrs={}):
#         if len(attrs) > 0:
#             html = "<%s %s>" % (
#             name, " ".join(["%s=\"%s\"" % (key, attrs[key]) for key in attrs.keys() if attrs[key] != ""]))
#         else:
#             html = "<%s>" % name
#         return self.__indent(html)
#
#
# class Bootstrap_Carousel:
#
#     def __init__(self, figure_list=[], name='carouselExample', args=None):
#         """Create a Bootstrap Carousel for a given list of figure files"""
#         self.flist = figure_list
#         self.name = name
#         self.args = args
#
#     def __repr__(self):
#         summary = []
#         summary.append("<bootstrap.carouselWithCaption>")
#         summary.append("Figures: %i" % len(self.flist))
#         summary.append("Name: %s" % self.name)
#         return "\n".join(summary)
#
#     def __html_carousel_btn(self, islide=0, active=False, target='carouselExample'):
#         attrs = {'type': "button",
#                  'data-bs-target': "#%s" % target,
#                  'data-bs-slide-to': "%i" % islide,
#                  'aria-label': "Slide %i" % int(islide + 1),
#                  'class': "active" if active else "",
#                  }
#         return HtmlHelper().cblock('button', attrs=attrs)
#
#     def __get_list_of_carousel_btn_html(self, this_flist, carouselName='carouselExample'):
#         html = []
#         for islide in np.arange(0, len(this_flist)):
#             html.append(self.__html_carousel_btn(islide, active=islide == 0, target=carouselName))
#         html = "\n".join(html)
#         return html
#
#     def __html_carousel_item(self, src='...', label='Slide label',
#                              description='Some representative placeholder content for the second slide.', active=False):
#         html = []
#         BH = lambda n: HtmlHelper(indent=n)
#         html.append(BH(n=0).block("div",
#                                   attrs={"class": "carousel-item %s" % ("active" if active else ""),
#                                          "data-bs-interval": 10}))
#         html.append(BH(n=2).block("img", attrs={'src': "{src}", 'class': 'd-block w-100', 'alt':''}))
#         html.append(BH(n=2).block("div", attrs={'class': 'carousel-caption d-none d-md-block'}))
#         html.append(BH(n=4).cblock("h5", content='{label}'))
#         html.append(BH(n=4).cblock("p", content='{description}'))
#         html.append(BH(n=2).block("/div"))
#         html.append(BH(n=0).block("/div"))
#         html = "\n".join(html).format(src=src, label=label, description=description)
#         return html
#
#     def __get_results_lnk(self, wmo, cyc, params):
#         this_args = parse_args(wmo, cyc)
#         this_request = {'args': {}}
#         for p in params:
#             if p == 'VEL':
#                 this_request['args']['velocity'] = params[p]
#                 this_args.velocity = params[p]
#             if p == 'NF':
#                 this_request['args']['nfloats'] = params[p]
#                 this_args.nfloats = params[p]
#             if p == 'CYCDUR':
#                 this_request['args']['cfg_cycle_duration'] = params[p]
#                 this_args.cfg_cycle_duration = params[p]
#             if p == 'PDPTH':
#                 this_request['args']['cfg_parking_depth'] = params[p]
#                 this_args.cfg_parking_depth = params[p]
#         results_lnk = url_for('.results', **request_opts_for_data(this_request, this_args))
#         return results_lnk
#
#     def __get_recap_lnk(self, wmo, cyc):
#         opts = request_opts_for_data(request, parse_args(wmo, cyc))
#         opts.pop('cyc')
#         results_lnk = url_for('.recap', **opts)
#         return results_lnk
#
#     def __get_list_of_carousel_items_html(self, this_flist):
#         html = []
#         for islide, figure_file in enumerate(this_flist):
#             params = read_params_from_path(figure_file, plist=['VEL', 'NF', 'CYCDUR', 'PDPTH'])
#             print(figure_file, params)
#             wmo = figure_file.replace("static/data/", "").split("/")[1]
#             cyc = figure_file.replace("static/data/", "").split("/")[2]
#             label = "Float %s - Cycle %s" % (wmo, cyc)
#             results_lnk = self.__get_results_lnk(wmo, cyc, params)
#             description = HtmlHelper().cblock("a", attrs={"href": results_lnk, "target": ""},
#                                               content="Check this cycle details")
#
#             # subsample_lnk = "/".join([request.url_root, 'recap', wmo]).replace("//recap", "/recap")
#             # subsample_lnk = "%s?velocity=%s&nfloats=%s" % (subsample_lnk, self.args.velocity, self.args.nfloats)
#             recap_lnk = self.__get_recap_lnk(wmo, cyc)
#             description = "%s / %s" % (HtmlHelper().cblock("a", attrs={"href": recap_lnk, "target": ""},
#                                           content="Swipe only this float"), description)
#
#             html.append(
#                 self.__html_carousel_item(src=figure_file, label=label, description=description, active=islide == 0))
#         html = "\n".join(html)
#         return html
#
#     @property
#     def html(self):
#         html = []
#         BH = lambda n: HtmlHelper(indent=n)
#
#         html.append(BH(n=0).block("div", attrs={"id": self.name,
#                                                 "class": "carousel carousel-dark slide",
#                                                 "data-bs-ride": "false"}))
#
#         html.append(BH(n=2).block("div", attrs={"class": "carousel-indicators"}))
#         html.append(self.__get_list_of_carousel_btn_html(self.flist, carouselName=self.name))
#         html.append(BH(n=2).block("/div"))
#
#         html.append(BH(n=2).block("div", attrs={"class": "carousel-inner"}))
#         html.append(self.__get_list_of_carousel_items_html(self.flist))
#         html.append(BH(n=2).block("/div"))
#
#         html.append(BH(n=2).block("button", attrs={"type": "button",
#                                                     "class": "carousel-control-prev",
#                                                     "data-bs-target": "#%s" % self.name,
#                                                     "data-bs-slide": "prev"}))
#         html.append(BH(n=4).cblock("span", attrs={"class": "carousel-control-prev-icon",
#                                                   "aria-hidden": "true"}))
#         html.append(BH(n=4).cblock("span", attrs={"class": "visually-hidden"}, content="Previous"))
#         html.append(BH(n=2).block("/button"))
#
#         html.append(BH(n=2).block("button", attrs={"type": "button",
#                                                     "class": "carousel-control-next",
#                                                     "data-bs-target": "#%s" % self.name,
#                                                     "data-bs-slide": "next"}))
#         html.append(BH(n=4).cblock("span", attrs={"class": "carousel-control-next-icon",
#                                                   "aria-hidden": "true"}))
#         html.append(BH(n=4).cblock("span", attrs={"class": "visually-hidden"}, content="Next"))
#         html.append(BH(n=2).block("/button"))
#
#         html.append(BH(n=0).block("/div"))
#         return "\n".join(html)
#
#
# class Bootstrap_Accordion:
#     def __init__(self, data=[], name='AccordionExample', args=None):
#         self.data = data
#         self.name = name
#         self.args = args
#
#     def __html_accordion_btn(self, txt="", collapsed=False, target=""):
#         attrs = {'type': "button",
#                  'data-bs-target': "#%s" % target,
#                  'data-bs-toggle': "collapse",
#                  'aria-expanded': "true",
#                  'aria-controls': "%s" % target,
#                  'class': "accordion-button %s" % ("collapsed" if collapsed else ""),
#                  }
#         return HtmlHelper().cblock("button", attrs=attrs, content=txt)
#         # return "<button %s>%s</button>" % (" ".join(
#         #     ["%s=\"%s\"" % (key, attrs[key]) for key in attrs.keys() if attrs[key] != ""]), txt)
#
#     def __html_accordion_item(self, title="", body="", itemID="", collapsed=False):
#         html = []
#         BH = lambda n: HtmlHelper(indent=n)
#         html.append(BH(0).block("div", attrs={"class": "accordion-item"}))
#         html.append(BH(2).block("h2", attrs={"class": "accordion-header", "id": "%s-heading" % itemID}))
#         html.append("    %s" % self.__html_accordion_btn(txt=title, collapsed=collapsed, target=itemID))
#         html.append(BH(2).block("/h2"))
#         html.append(BH(2).block("div", attrs={"id": "%s" % itemID,
#                                               "class": "accordion-collapse collapse %s" % ("show" if not collapsed else ""),
#                                               "aria-labelledby": "%s-heading" % itemID}))
#         html.append(BH(4).block("div", attrs={"class": "accordion-body"}))
#         html.append("      %s" % body)
#         html.append(BH(4).block("/div"))
#         html.append(BH(2).block("/div"))
#         html.append(BH(0).block("/div"))
#         return "\n".join(html)
#
#     @property
#     def html(self):
#         html = []
#         html.append(HtmlHelper().block("div", attrs={"class": "accordion w-100", "id": self.name}))
#         for ii, item in enumerate(self.data):
#             item_html = self.__html_accordion_item(title=item['title'],
#                                                    body=item['body'],
#                                                    itemID="%s-item%i" % (self.name, ii),
#                                                    collapsed=ii != 0)
#             html.append(item_html)
#         html.append(HtmlHelper().block("/div"))
#         return "\n".join(html)
#
#
# class Bootstrap_Figure:
#     def __init__(self, src=None, alt="", caption=""):
#         """Return a Boostrap Figure html"""
#         self.src = src
#         self.alt = alt
#         self.caption = caption
#
#     @property
#     def html(self):
#         html = []
#         # html.append("<figure class=\"figure\">")
#         # html.append("  <img src=\"{src}\" class=\"figure-img img-fluid rounded\" alt=\"{alt}\">")
#         # html.append("  <figcaption class=\"figure-caption\">{caption}</figcaption>")
#         # html.append("</figure>")
#         html.append(HtmlHelper(indent=0).block("figure", attrs={"class": "figure"}))
#         html.append(HtmlHelper(indent=2).block("img", attrs={"class": "figure-img img-fluid rounded",
#                                                              "src": "{src}",
#                                                              "alt": "{alt}"}))
#         html.append(HtmlHelper(indent=2).cblock("figcaption", attrs={"class": "figure-caption"}, content="{caption}"))
#         html.append(HtmlHelper(indent=0).block("/figure"))
#         html = "\n".join(html).format(src=self.src, alt=self.alt, caption=self.caption)
#         return html
