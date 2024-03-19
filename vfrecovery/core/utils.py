import pandas as pd
from typing import List
from argopy import ArgoIndex
import argopy.plot as argoplot

from vfrecovery.json import Profile, MetaData


def ArgoIndex2df(a_wmo, a_cyc) -> pd.DataFrame:
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
    df = ArgoIndex(host=host).search_wmo_cyc(a_wmo, a_cyc).to_dataframe()
    return df


def df_obs2jsProfile(df_obs) -> List[Profile]:
    Plist = Profile.from_ArgoIndex(df_obs)
    for P in Plist:
        P.description = "Observed Argo profile"
    return Plist


def ArgoIndex2JsProfile(a_wmo, a_cyc) -> List[Profile]:
    """Retrieve WMO/CYC Argo index entries as a list of :class:`vfrecovery.json.Profile`

    Parameters
    ----------
    wmo: int
    cyc: Union(int, List)

    Returns
    -------
    :class:`vfrecovery.json.Profile`
    """
    df_obs = ArgoIndex2df(a_wmo, a_cyc)
    return df_obs2jsProfile(df_obs)


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
