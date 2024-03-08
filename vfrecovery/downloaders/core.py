import pandas as pd
import os
import xarray as xr

from . import Glorys, Armor3d
# import logging


# log = logging.getLogger("vfrecovery.download.core")


def get_velocity_field(a_box, a_date, n_days=1, output='.', dataset='ARMOR3D'):
    """Return the velocity field as an :class:xr.Dataset, download if needed

    Parameters
    ----------
    a_box
    a_date
    n_days
    output
    dataset
    """
    def get_velocity_filename(dataset, n_days):
        download_date = pd.to_datetime('now', utc='now').strftime("%Y%m%d")
        fname = os.path.join(output, 'velocity_%s_%idays_%s.nc' % (dataset, n_days, download_date))
        return fname

    velocity_file = get_velocity_filename(dataset, n_days)
    if not os.path.exists(velocity_file):
        # Define Data loader:
        loader = Armor3d if dataset == 'ARMOR3D' else Glorys
        loader = loader(a_box, a_date, n_days=n_days)
        # puts(str(loader), color=COLORS.magenta)

        # Load data from Copernicus Marine Data store:
        ds = loader.to_xarray()

        # Save on file for later re-used:
        ds.to_netcdf(velocity_file)
    else:
        ds = xr.open_dataset(velocity_file)

    return ds, velocity_file
