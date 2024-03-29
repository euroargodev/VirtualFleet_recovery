import pandas as pd
import os
import xarray as xr

from . import Glorys, Armor3d
from vfrecovery.core.utils import pp_obj


def get_velocity_field(a_box, a_date, n_days=1, output='.', dataset='ARMOR3D', logger=None, lazy=True) -> tuple:
    """Return the velocity field as an :class:`xr.Dataset`, force download/save if not lazy

    Parameters
    ----------
    a_box
    a_date
    n_days
    output
    dataset
    logger
    lazy

    Retuns
    ------
    tuple
    """

    access_date = pd.to_datetime('now', utc='now').strftime("%Y%m%d")

    def get_velocity_filename(dataset, n_days):
        fname = os.path.join(output, 'velocity_%s_%idays_%s.nc' % (dataset, n_days, access_date))
        return fname

    velocity_file = get_velocity_filename(dataset, n_days)

    if not os.path.exists(velocity_file):
        new = True
        # Define Data loader:
        loader = Armor3d if dataset == 'ARMOR3D' else Glorys

        # Make an instance
        # (we add a 1-day security delay at the beginning to make sure that we have velocity at the deployment time)
        loader = loader(a_box, a_date - pd.Timedelta(1, 'D'), n_days=n_days+1, logger=logger)

        # Load data from the Copernicus Marine Data store:
        ds = loader.to_xarray()  # Lazy by default
        if logger is not None:
            logger.debug(pp_obj(loader))

        # Save on file for later re-used:
        # (this can take a while and is often longer than the lazy mode !)
        if not lazy:
            if logger is not None:
                logger.debug("Saving velocity on file for later re-used")
            ds.to_netcdf(velocity_file)

    else:
        new = False
        ds = xr.open_dataset(velocity_file)

    ds.attrs['access_date'] = access_date

    return ds, velocity_file, new
