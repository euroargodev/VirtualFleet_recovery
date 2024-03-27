import logging
import json
from typing import Union
from argopy.utils import is_wmo, is_cyc, check_cyc, check_wmo
import argopy.plot as argoplot
from argopy.errors import DataNotFound
from argopy import ArgoIndex

from .utils import ArgoIndex2df_obs

root_logger = logging.getLogger("vfrecovery_root_logger")


def describe_obs(wmo, cyc):

    # Validate arguments:
    assert is_wmo(wmo)
    wmo = check_wmo(wmo)[0]
    if cyc is not None:
        assert is_cyc(cyc)
        cyc = check_cyc(cyc)[0]

    #
    url = argoplot.dashboard(wmo, url_only=True)
    txt = "You can check this float dashboard while we search for float profiles in the index: %s" % url
    root_logger.info(txt)

    # Load observed float profiles index:
    host = "https://data-argo.ifremer.fr"
    # host = "/home/ref-argo/gdac" if os.uname()[0] == 'Darwin' else "https://data-argo.ifremer.fr"
    # host = "/home/ref-argo/gdac" if not os.uname()[0] == 'Darwin' else "~/data/ARGO"
    idx = ArgoIndex(host=host)
    if cyc is not None:
        idx.search_wmo_cyc(wmo, cyc)
    else:
        idx.search_wmo(wmo)

    df = idx.to_dataframe()
    df = df.sort_values(by='date')
    root_logger.info("\n%s" % df.to_string(max_colwidth=15))

    # output = {'wmo': wmo, 'cyc': cyc}
    # json_dump = json.dumps(
    #     output, sort_keys=False, indent=2
    # )
    # return json_dump


def describe_function(
        wmo: int,
        cyc: Union[int, None],
        target: str,
        log_level: str,
) -> str:
    if log_level == "QUIET":
        root_logger.disabled = True
        log_level = "CRITICAL"
    root_logger.setLevel(level=getattr(logging, log_level.upper()))

    if target == 'obs':
        describe_obs(wmo, cyc)