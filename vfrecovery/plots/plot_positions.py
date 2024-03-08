import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import numpy as np

from .utils import get_HBOX, map_add_features, map_add_profiles, save_figurefile


def figure_positions(this_args, vel, df_sim, df_plan, this_profile, cfg, wmo, cyc, vel_name,
                     dd=1, save_figure=False, workdir='.'):
    # log.debug("Starts figure_positions")
    ebox = get_HBOX(df_sim, dd=dd)
    nfloats = df_plan.shape[0]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 7), dpi=120,
                           subplot_kw={'projection': ccrs.PlateCarree()},
                           sharex=True, sharey=True)
    ax = ax.flatten()

    for ix in [0, 1, 2]:
        ax[ix].set_extent(ebox)
        ax[ix] = map_add_features(ax[ix])

        v = vel.field.isel(time=0).interp(depth=cfg.mission['parking_depth']).plot.quiver(x="longitude",
                                                                                   y="latitude",
                                                                                   u=vel.var['U'],
                                                                                   v=vel.var['V'],
                                                                                   ax=ax[ix],
                                                                                   color='grey',
                                                                                   alpha=0.5,
                                                                                   add_guide=False)

        ax[ix].plot(df_sim['deploy_lon'], df_sim['deploy_lat'], '.',
                    markersize=3, color='grey', alpha=0.1, markeredgecolor=None, zorder=0)
        if ix == 0:
            title = 'Velocity field at %0.2fm and deployment plan' % cfg.mission['parking_depth']
            v.set_alpha(1)
            # v.set_color('black')
        elif ix == 1:
            x, y, c = df_sim['longitude'], df_sim['latitude'], df_sim['cyc']
            title = 'Final float positions'
            # sc = ax[ix].plot(x, y, '.', markersize=3, color='cyan', alpha=0.9, markeredgecolor=None)
            sc = ax[ix].scatter(x, y, c=c, s=3, alpha=0.9, edgecolors=None)
        elif ix == 2:
            x, y, c = df_sim['rel_lon'], df_sim['rel_lat'], df_sim['cyc']
            title = 'Final floats position relative to last float position'
            # sc = ax[ix].plot(x, y, '.', markersize=3, color='cyan', alpha=0.9, markeredgecolor=None)
            sc = ax[ix].scatter(x, y, c=c, s=3, alpha=0.9, edgecolors=None)

        ax[ix] = map_add_profiles(ax[ix], this_profile)
        ax[ix].set_title(title)

    fig.suptitle("VirtualFleet recovery prediction for WMO %i: starting from cycle %i, predicting cycle %s\n%s" %
                 (wmo, cyc[0], cyc[1:], get_cfg_str(cfg)), fontsize=15)
    plt.tight_layout()
    if save_figure:
        save_figurefile(fig, "vfrecov_positions_%s" % get_sim_suffix(this_args, cfg), workdir)
    return fig, ax

