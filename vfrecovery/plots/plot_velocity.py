import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import numpy as np

from .utils import map_add_profiles, map_add_features, save_figurefile


def figure_velocity(box,
                       vel, vel_name, this_profile, wmo, cyc,
                       save_figure=False, workdir='.'):
    """

    Parameters
    ----------
    box

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(box)
    ax = map_add_features(ax)
    ax = map_add_profiles(ax, this_profile)

    vel.field.isel(time=0, depth=0).plot.quiver(x="longitude", y="latitude",
                                                u=vel.var['U'], v=vel.var['V'], ax=ax, color='grey', alpha=0.5,
                                          add_guide=False)

    txt = "starting from cycle %i, predicting cycle %i" % (cyc[0], cyc[1])
    ax.set_title(
        "VirtualFleet recovery system for WMO %i: %s\n"
        "%s velocity snapshot to illustrate the simulation domain\n"
        "Vectors: Velocity field at z=%0.2fm, t=%s" %
        (wmo, txt, vel_name, vel.field['depth'][0].values[np.newaxis][0],
        pd.to_datetime(vel.field['time'][0].values).strftime("%Y/%m/%d %H:%M")), fontsize=15)

    plt.tight_layout()
    if save_figure:
        save_figurefile(fig, 'vfrecov_velocity_%s' % vel_name, workdir)
    return fig, ax
