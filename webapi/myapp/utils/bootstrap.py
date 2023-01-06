"""

Module with class to be used to produce HTML for Boostrap components:
- Bootstrap_Figure: https://getbootstrap.com/docs/5.3/content/figures/
- Bootstrap_Accordion: https://getbootstrap.com/docs/5.3/components/accordion/
- Bootstrap_Carousel: https://getbootstrap.com/docs/4.0/components/carousel/

"""
import numpy as np
from dominate.tags import button, div, h2, h5, p, a, span
from dominate.tags import img, figure, figcaption
from dominate.util import raw
from abc import ABC


class Bootstrap_Figure(ABC):
    """Return a Boostrap figure

    Based on: https://getbootstrap.com/docs/5.3/content/figures/

    HTML pattern of the figure component:
    ```html
        <figure class="figure">
          <img src="..." class="figure-img img-fluid rounded" alt="...">
          <figcaption class="figure-caption text-end">A caption for the above image.</figcaption>
        </figure>
    ```
    """
    def __init__(self, src=None, alt="", caption=""):
        """Return a Boostrap Figure html

        Parameters
        ----------
        src: str
            Path the image file
        alt: str, optional
            Alternative text to the image
        caption: str, optional
            Text to insert as a caption to the figure

        Examples
        --------
        >>> Bootstrap_Figure(src='logo-virtual-fleet-recovery.png', caption='Hello').html
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
        return f.render()


class Bootstrap_Accordion(ABC):
    """Return a Boostrap accordion

    Based on: https://getbootstrap.com/docs/5.3/components/accordion/

    HTML pattern of an accordion:
    ```html
        <div class="accordion" id="accordionPanelsStayOpenExample">
          <div class="accordion-item">
            <h2 class="accordion-header" id="panelsStayOpen-headingOne">
              <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#panelsStayOpen-collapseOne" aria-expanded="true" aria-controls="panelsStayOpen-collapseOne">
                Accordion Item #1
              </button>
            </h2>
            <div id="panelsStayOpen-collapseOne" class="accordion-collapse collapse show" aria-labelledby="panelsStayOpen-headingOne">
              <div class="accordion-body">
                <strong>This is the first item's accordion body.</strong>
              </div>
            </div>
          </div>
          <div class="accordion-item">
            <h2 class="accordion-header" id="panelsStayOpen-headingTwo">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#panelsStayOpen-collapseTwo" aria-expanded="false" aria-controls="panelsStayOpen-collapseTwo">
                Accordion Item #2
              </button>
            </h2>
            <div id="panelsStayOpen-collapseTwo" class="accordion-collapse collapse" aria-labelledby="panelsStayOpen-headingTwo">
              <div class="accordion-body">
                <strong>This is the second item's accordion body.</strong>
              </div>
            </div>
          </div>
          <div class="accordion-item">
            <h2 class="accordion-header" id="panelsStayOpen-headingThree">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#panelsStayOpen-collapseThree" aria-expanded="false" aria-controls="panelsStayOpen-collapseThree">
                Accordion Item #3
              </button>
            </h2>
            <div id="panelsStayOpen-collapseThree" class="accordion-collapse collapse" aria-labelledby="panelsStayOpen-headingThree">
              <div class="accordion-body">
                <strong>This is the third item's accordion body.</strong>
              </div>
            </div>
          </div>
        </div>
    ```
    """
    def __init__(self, data=[], id='AccordionExample'):
        """Return a Bootstrap accordion html

        Parameters
        ----------
        data: list of dict with ``title`` and ``description`` keys
            List of accordion panels data. For each panel (i.e. an item of the ``data`` list), we use the ``title`` value
            to populate the accordion header, and the ``description`` value for the accordion body.
        id: str, optional, default='AccordionExample'
            HTML id of the accordion

        Examples
        --------
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
        html = Bootstrap_Accordion(data=data, id='Figures').html
        """
        self.data = data
        self.id = id

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
        d += h2(self._html_accordion_btn(txt=title, collapsed=collapsed, target=itemID),
                cls="accordion-header",
                id="%s-heading" % itemID)
        d += div(div(raw(body), cls="accordion-body"),
                 id="%s" % itemID,
                 cls="accordion-collapse collapse %s" % ("show" if not collapsed else ""),
                 aria_labelledby="%s-heading" % itemID)
        return d

    @property
    def html(self):
        code = div(cls="accordion w-100", id=self.id)
        for ii, item in enumerate(self.data):
            with code:
                self._html_accordion_item(title=item['title'],
                                          body=item['body'],
                                          itemID="%s-item%i" % (self.id, ii),
                                          collapsed=ii != 0)
        return code.render()


class Bootstrap_Carousel(ABC):
    """Return a Boostrap carousel, a slideshow component for cycling through elements—images or slides of text—like a carousel.

    Based on: https://getbootstrap.com/docs/4.0/components/carousel/

    HTML pattern of a carousel:
    ```html
        <div id="carouselExample" class="carousel slide">

          <div class="carousel-indicators">
            <button type="button" data-bs-target="#carouselExample" data-bs-slide-to="0" class="active" aria-current="true" aria-label="Slide 1"></button>
            <button type="button" data-bs-target="#carouselExample" data-bs-slide-to="1" aria-label="Slide 2"></button>
            <button type="button" data-bs-target="#carouselExample" data-bs-slide-to="2" aria-label="Slide 3"></button>
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


          <button class="carousel-control-prev" type="button" data-bs-target="#carouselExample" data-bs-slide="prev">
            <span class="carousel-control-prev-icon" aria-hidden="true"></span>
            <span class="visually-hidden">Previous</span>
          </button>
          <button class="carousel-control-next" type="button" data-bs-target="#carouselExample" data-bs-slide="next">
            <span class="carousel-control-next-icon" aria-hidden="true"></span>
            <span class="visually-hidden">Next</span>
          </button>

        </div>
    ```
    """

    def __init__(self, figure_list=[], id='carouselExample', label=None, description=None):
        """Create a Bootstrap Carousel for a given list of figure files

        Instantiate with a list of files and access the `html` instance property to get the code to insert.

        Parameters
        ----------
        figure_list: list of str
            The list of figure files to insert in the carousel
        id: str, optional, default 'carouselExample'
            HTML id of the carousel
        label: function, optional
            A function called on each slide with the number and figure file as arguments and that return a label string
            to insert on each slide caption.
        description: function, optional
            A function called on each slide with the number and figure file as arguments and that return a description
            string to insert on each slide caption.
        """
        self.flist = figure_list
        self.id = id
        self.get_label = self._get_label if label is None else label
        self.get_description = self._get_description if description is None else description

    def __repr__(self):
        summary = []
        summary.append("<bootstrap.carouselWithCaption>")
        summary.append("Figures: %i" % len(self.flist))
        summary.append("ID: %s" % self.id)
        return "\n".join(summary)

    def _get_label(self, slide_number, figure_file):
        return "Figure %i" % (slide_number + 1)

    def _get_description(self, slide_number, figure_file):
        return "File: %s" % figure_file

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

    def get_carousel_item(self,
                          src='...',
                          label='Slide label',
                          description='Some representative placeholder content for this slide.',
                          active=False):
        d = div(cls="carousel-item %s" % ("active" if active else ""), data_bs_interval=10)
        d += img(src=src, cls='d-block w-100', alt='')
        d += div([h5(label), p(raw(description))], cls="carousel-caption d-none d-md-block")
        return d

    def get_list_of_carousel_items(self):
        d = div(cls="carousel-inner")
        for islide, figure_file in enumerate(self.flist):
            label = self.get_label(islide, figure_file)
            description = self.get_description(islide, figure_file)
            d += self.get_carousel_item(src=figure_file, label=label, description=description, active=islide == 0)
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
        for b in self.get_carousel_controls():
            d += b
        return d.render()

