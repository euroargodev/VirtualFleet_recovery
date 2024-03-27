import pandas as pd
import numpy as np
from typing import List
from argopy import ArgoIndex
import argopy.plot as argoplot
from argopy.errors import DataNotFound

from vfrecovery.json import Profile, MetaData

pp_obj = lambda x: "\n%s" % "\n".join(["\t%s" % line for line in x.__repr__().split("\n")])


def ArgoIndex2df_obs(a_wmo, a_cyc, cache:bool=False, cachedir:str='.') -> pd.DataFrame:
    """Retrieve WMO/CYC Argo index entries as :class:`pd.DataFrame`

    Parameters
    ----------
    wmo: int
    cyc: Union(int, List)

    Returns
    -------
    :class:`pd.DataFrame`
    """
    host = "https://data-argo.ifremer.fr"
    # host = "/home/ref-argo/gdac" if os.uname()[0] == 'Darwin' else "https://data-argo.ifremer.fr"
    # host = "/home/ref-argo/gdac" if not os.uname()[0] == 'Darwin' else "~/data/ARGO"
    idx = ArgoIndex(host=host, cache=cache, cachedir=cachedir).search_wmo_cyc(a_wmo, a_cyc)
    if idx.N_MATCH == 0:
        raise DataNotFound("This float has no cycle %i usable as initial conditions for a simulation of %i" % (a_cyc[0], a_cyc[1]))
    else:
        df = idx.to_dataframe()
    df = df.sort_values(by='date')
    return df


def df_obs2jsProfile(df_obs) -> List[Profile]:
    Plist = Profile.from_ArgoIndex(df_obs)
    for P in Plist:
        P.description = "Observed Argo profile"
        P.location.description = None
    return Plist


def ArgoIndex2jsProfile(a_wmo, a_cyc, cache:bool=False, cachedir:str='.') -> List[Profile]:
    """Retrieve WMO/CYC Argo index entries as a list of :class:`vfrecovery.json.Profile`

    Parameters
    ----------
    wmo: int
    cyc: Union(int, List)

    Returns
    -------
    :class:`vfrecovery.json.Profile`
    """
    df_obs = ArgoIndex2df_obs(a_wmo, a_cyc, cache=cache, cachedir=cachedir)
    return df_obs2jsProfile(df_obs), df_obs


def get_simulation_suffix(md: MetaData) -> str:
    """Compose a simulation unique ID for output files"""
    # suf = '%s_%i' % (this_args.velocity, this_args.nfloats)
    suf = 'VEL%s_NFL%i_CYT%i_PKD%i_PFD%i_FSD%i' % (md.velocity_field,
                                                         md.n_floats,
                                                         int(md.vfconfig.mission['cycle_duration']),
                                                         int(md.vfconfig.mission['parking_depth']),
                                                         int(md.vfconfig.mission['profile_depth']),
                                                         int(md.vfconfig.mission['reco_free_surface_drift']))
    return suf


def get_domain(Plist, size):
    c = [np.mean([p.location.longitude for p in Plist]), np.mean([p.location.latitude for p in Plist])]
    domain = [c[0] - size / 2, c[0] + size / 2,
              c[1] - size / 2, c[1] + size / 2]
    domain = [np.round(d, 3) for d in domain]
    return domain, c