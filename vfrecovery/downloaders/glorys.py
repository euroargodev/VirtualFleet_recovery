import os
import numpy as np
import glob
import pandas as pd
import xarray as xr
import copernicusmarine
import logging


logger = logging.getLogger("vfrecovery.downloaders")

class default_logger:

    def __init__(self, txt, log_level):
        """Log text"""
        getattr(logger, log_level.lower())(txt)

    @staticmethod
    def info(txt) -> 'default_logger':
        return default_logger(txt, 'INFO')

    @staticmethod
    def debug(txt) -> 'default_logger':
        return default_logger(txt, 'DEBUG')

    @staticmethod
    def warning(txt) -> 'default_logger':
        return default_logger(txt, 'WARNING')

    @staticmethod
    def error(txt) -> 'default_logger':
        return default_logger(txt, 'ERROR')


def get_glorys_forecast_from_datarmor(a_box, a_start_date, n_days=1):
    """Load Datarmor Global Ocean 1/12° Physics Analysis and Forecast updated Daily

    Fields: daily, from 2020-11-25T12:00 to 'now' + 5 days
    Src: /home/ref-ocean-model-public/multiparameter/physic/global/cmems/global-analysis-forecast-phy-001-024
    Info: https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSISFORECAST_PHY_001_024/INFORMATION

    Parameters
    ----------
    a_box
    a_start_date
    n_days
    """
    def get_forecast_files(a_date, n_days=1):
        file_list = []
        for n in range(0, n_days):
            t = a_date + pd.Timedelta(n, 'D')
            p = os.path.join(src, "%i" % t.year, "%0.3d" % t.day_of_year)
            # print(p, os.path.exists(p))
            if os.path.exists(p):
                file_list.append(sorted(glob.glob(os.path.join(p, "*.nc")))[0])
        return file_list

    def preprocess(this_ds):
        idpt = np.argwhere(this_ds['depth'].values > 2000)[0][0]
        ilon = np.argwhere(this_ds['longitude'].values >= a_box[0])[0][0], \
               np.argwhere(this_ds['longitude'].values >= a_box[1])[0][0]
        ilat = np.argwhere(this_ds['latitude'].values >= a_box[2])[0][0], \
               np.argwhere(this_ds['latitude'].values >= a_box[3])[0][0]
        this_ds = this_ds.isel({'depth': range(0, idpt),
                                'longitude': range(ilon[0], ilon[1]),
                                'latitude': range(ilat[0], ilat[1])})
        return this_ds

    root = "/home/ref-ocean-model-public" if not os.uname()[0] == 'Darwin' else "/Volumes/MODEL-PUBLIC/"
    src = os.path.join(root, "multiparameter/physic/global/cmems/global-analysis-forecast-phy-001-024")
    # puts("\t%s" % src, color=COLORS.green)
    flist = get_forecast_files(a_start_date, n_days=n_days)
    if len(flist) == 0:
        raise ValueError("This float cycle is too old for this velocity field.")
    glorys = xr.open_mfdataset(flist, preprocess=preprocess, combine='nested', concat_dim='time', parallel=True)
    #
    return glorys


class Glorys:
    """Global Ocean 1/12° Physics Re-Analysis or Forecast

    If start_date + n_days <= 2021-01-09:
        Delivers the multi-year reprocessed (REP) daily data
        https://resources.marine.copernicus.eu/product-detail/GLOBAL_MULTIYEAR_PHY_001_030

    otherwise:
        Delivers the near-real-time (NRT) Analysis and Forecast daily data
        https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSISFORECAST_PHY_001_024

    Examples
    --------
    >>> Glorys([-25, -13, 6.5, 13], pd.to_datetime('20091130', utc=True)).to_xarray()
    >>> Glorys([-25, -13, 6.5, 13], pd.to_datetime('20231121', utc=True), n_days=10).to_xarray()

    """

    def __init__(self, box, start_date, n_days=1, max_depth=2500, **kwargs):
        """
        Parameters
        ----------
        box: list(float)
            Define domain to load: [lon_min, lon_max, lat_min, lat_max]
        start_date: :class:`pandas.Timestamp`
            Starting date of the time series to load.
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
        if start_date + dt <= pd.to_datetime('2021-01-09', utc=True):
            self._loader = self._get_reanalysis
            self.dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
        else:
            self._loader = self._get_forecast
            self.dataset_id = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"

        self.logger = kwargs['logger'] if 'logger' in kwargs else default_logger
        self.overwrite_metadata_cache = kwargs['overwrite_metadata_cache'] if 'overwrite_metadata_cache' in kwargs else False
        self.disable_progress_bar = kwargs['disable_progress_bar'] if 'disable_progress_bar' in kwargs else False

    def _get_this(self, dataset_id, dates):
        ds = copernicusmarine.open_dataset(
            dataset_id=dataset_id,
            minimum_longitude=self.box[0],
            maximum_longitude=self.box[1],
            minimum_latitude=self.box[2],
            maximum_latitude=self.box[3],
            maximum_depth=self.max_depth,
            start_datetime=dates[0].strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=dates[1].strftime("%Y-%m-%dT%H:%M:%S"),
            variables=['uo', 'vo'],
            disable_progress_bar=self.disable_progress_bar,
            overwrite_metadata_cache=self.overwrite_metadata_cache,
        )
        return ds

    def _get_forecast(self):
        """
        Returns
        -------
        :class:`xarray.dataset`
        """
        start_date = self.start_date
        if self.n_days == 1:
            end_date = start_date
        else:
            end_date = start_date + pd.Timedelta(self.n_days + 1, 'D')
        return self._get_this(self.dataset_id, [start_date, end_date])

    def _get_reanalysis(self):
        """
        Returns
        -------
        :class:`xarray.dataset`
        """
        start_date = self.start_date
        if self.n_days == 1:
            end_date = start_date
        else:
            end_date = self.start_date + pd.Timedelta(self.n_days + 1, 'D')
        return self._get_this(self.dataset_id, [start_date, end_date])

    def to_xarray(self):
        """ Load and return data as a :class:`xarray.dataset`
        Returns
        -------
        :class:`xarray.dataset`
        """
        return self._loader()

    def __repr__(self):
        summary = ["<CopernicusMarineData.Loader><Glorys>"]
        summary.append("dataset_id: %s" % self.dataset_id)
        summary.append("Starting date: %s" % self.start_date)
        summary.append("N days: %s" % self.n_days)
        summary.append("Domain: %s" % self.box)
        summary.append("Max depth (m): %s" % self.max_depth)
        return "\n".join(summary)

