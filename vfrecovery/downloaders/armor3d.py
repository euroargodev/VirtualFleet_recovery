import pandas as pd
import copernicusmarine


class Armor3d:
    """Global Ocean 1/4° Multi Observation Product ARMOR3D

    Product description:
    https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012

    If start_date + n_days <= 2022-12-28:
        Delivers the multi-year reprocessed (REP) weekly data

    otherwise:
        Delivers the near-real-time (NRT) weekly data

    Examples
    --------
    >>> Armor3d([-25, -13, 6.5, 13], pd.to_datetime('20091130', utc=True)).to_xarray()
    >>> Armor3d([-25, -13, 6.5, 13], pd.to_datetime('20231121', utc=True), n_days=10).to_xarray()

    """

    def __init__(self, box, start_date, n_days=1, max_depth=2500):
        """
        Parameters
        ----------
        box: list(float)
            Define domain to load: [lon_min, lon_max, lat_min, lat_max]
        start_date: :class:`pandas.Timestamp`
            Starting date of the time series to load. Since ARMOR3D is weekly, the effective starting
            date will be the first weekly period including the user-defined ``start_date``
        n_days: int (default=1)
            Number of days to load data for.
        max_depth: float (default=2500)
            Maximum depth levels to load data for.
        """
        self.box = box
        self.start_date = start_date
        self.n_days = n_days
        self.max_depth = max_depth

        dt = pd.Timedelta(n_days, 'D') if n_days > 1 else pd.Timedelta(0, 'D')
        if start_date + dt <= pd.to_datetime('2022-12-28', utc=True):
            self._loader = self._get_rep
            self.dataset_id = "dataset-armor-3d-rep-weekly"
            self.time_axis = pd.Series(pd.date_range('19930106', '20221228', freq='7D').tz_localize("UTC"))
        else:
            self._loader = self._get_nrt
            self.dataset_id = "dataset-armor-3d-nrt-weekly"
            self.time_axis = pd.Series(
                pd.date_range('20190102', pd.to_datetime('now', utc=True).strftime("%Y%m%d"), freq='7D').tz_localize(
                    "UTC")[0:-1])

        if start_date < self.time_axis.iloc[0]:
            raise ValueError('Date out of bounds')
        elif start_date + dt > self.time_axis.iloc[-1]:
            raise ValueError('Date out of bounds, %s > %s' % (
                start_date + dt, self.time_axis.iloc[-1]))

    def _get_this(self, dataset_id):
        start_date = self.time_axis[self.time_axis <= self.start_date].iloc[-1]
        if self.n_days == 1:
            end_date = start_date
        else:
            end_date = \
            self.time_axis[self.time_axis <= self.start_date + (self.n_days + 7) * pd.Timedelta(1, 'D')].iloc[-1]

        ds = copernicusmarine.open_dataset(
            dataset_id=dataset_id,
            minimum_longitude=self.box[0],
            maximum_longitude=self.box[1],
            minimum_latitude=self.box[2],
            maximum_latitude=self.box[3],
            maximum_depth=self.max_depth,
            start_datetime=start_date.strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=end_date.strftime("%Y-%m-%dT%H:%M:%S"),
            variables=['ugo', 'vgo']
        )
        return ds

    def _get_rep(self):
        """multi-year reprocessed (REP) weekly data

        Returns
        -------
        :class:xarray.dataset
        """
        return self._get_this(self.dataset_id)

    def _get_nrt(self):
        """near-real-time (NRT) weekly data

        Returns
        -------
        :class:xarray.dataset
        """
        return self._get_this(self.dataset_id)

    def to_xarray(self):
        """Load and return data as a :class:`xarray.dataset`

        Returns
        -------
        :class:xarray.dataset
        """
        return self._loader()

    def __repr__(self):
        summary = ["<CopernicusMarineData.Loader><Armor3D>"]
        summary.append("dataset_id: %s" % self.dataset_id)
        summary.append("First day: %s" % self.start_date)
        summary.append("N days: %s" % self.n_days)
        summary.append("Domain: %s" % self.box)
        summary.append("Max depth (m): %s" % self.max_depth)
        return "\n".join(summary)
