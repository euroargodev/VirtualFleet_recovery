import os
import numpy as np
import argopy.plot as argoplot
from pathlib import Path


def save_figurefile(this_fig, a_name, folder: Path = Path('.')):
    """
    Parameters
    ----------
    this_fig
    a_name
    Path

    Returns
    -------
    path
    """
    figname = folder.joinpath("%s.png" % a_name)
    this_fig.savefig(figname)
    return figname

def map_add_profiles(this_ax, this_profile):
    """

    Parameters
    ----------
    this_ax

    Returns
    -------
    this_ax
    """
    this_ax.plot(this_profile['longitude'][0], this_profile['latitude'][0], 'k.', markersize=10, markeredgecolor='w')
    if this_profile.shape[0] > 1:
        this_ax.plot(this_profile['longitude'][1], this_profile['latitude'][1], 'r.', markersize=10, markeredgecolor='w')
        this_ax.arrow(this_profile['longitude'][0],
                 this_profile['latitude'][0],
                 this_profile['longitude'][1] - this_profile['longitude'][0],
                 this_profile['latitude'][1] - this_profile['latitude'][0],
                 length_includes_head=True, fc='k', ec='k', head_width=0.025, zorder=10)

    return this_ax


def map_add_features(this_ax):
    """

    Parameters
    ----------
    this_ax

    Returns
    -------
    this_ax
    """
    argoplot.utils.latlongrid(this_ax)
    this_ax.add_feature(argoplot.utils.land_feature, edgecolor="black")
    return this_ax


def map_add_cyc_nb(this_ax, this_df, lon='lon', lat='lat', cyc='cyc', pos='bt', fs=6, color='black'):
    """ Add cycle number labels next to axis

    Parameters
    ----------
    ax
    df

    Returns
    -------
    list of text label
    """
    t = []
    if pos == 'bt':
        ha, va, label = 'center', 'top', "\n{}".format
    if pos == 'tp':
        ha, va, label = 'center', 'bottom', "{}\n".format
    for irow, row in this_df.iterrows():
        this_t = this_ax.text(row[lon], row[lat], label(int(row[cyc])), ha=ha, va=va, fontsize=fs, color=color)
        t.append(this_t)
    return t


def get_HBOX(df_sim, dd=1):
    """

    Parameters
    ----------
    dd: how much to extend maps outward the deployment 'box'

    Returns
    -------
    list
    """
    rx = df_sim['deploy_lon'].max() - df_sim['deploy_lon'].min()
    ry = df_sim['deploy_lat'].max() - df_sim['deploy_lat'].min()
    lonc, latc = df_sim['deploy_lon'].mean(), df_sim['deploy_lat'].mean()
    box = [lonc - rx / 2, lonc + rx / 2, latc - ry / 2, latc + ry / 2]
    ebox = [box[i] + [-dd, dd, -dd, dd][i] for i in range(0, 4)]  # Extended 'box'

    return ebox


def get_EBOX(df_sim, df_plan, this_profile, s=1):
    """Get a box for maps

    Use all data positions from DF_SIM to make sure all points are visible
    Extend the domain by a 's' scaling factor of the deployment plan domain

    Parameters
    ----------
    s: float, default:1

    Returns
    -------
    list
    """
    box = [np.min([df_sim['deploy_lon'].min(), df_sim['longitude'].min(), df_sim['rel_lon'].min(), this_profile['longitude'].min()]),
       np.max([df_sim['deploy_lon'].max(), df_sim['longitude'].max(), df_sim['rel_lon'].max(), this_profile['longitude'].max()]),
       np.min([df_sim['deploy_lat'].min(), df_sim['latitude'].min(), df_sim['rel_lat'].min(), this_profile['latitude'].min()]),
       np.max([df_sim['deploy_lat'].max(), df_sim['latitude'].max(), df_sim['rel_lat'].max(), this_profile['latitude'].max()])]
    rx, ry = df_plan['longitude'].max() - df_plan['longitude'].min(), df_plan['latitude'].max() - df_plan['latitude'].min()
    r = np.min([rx, ry])
    ebox = [box[0]-s*r, box[1]+s*r, box[2]-s*r, box[3]+s*r]

    return ebox

