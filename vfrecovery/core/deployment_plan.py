import numpy as np
import pandas as pd
from vfrecovery.json import Profile


def setup_deployment_plan(P: Profile, nfloats: int = 120):
    # We will deploy a collection of virtual floats that are located around the real float with random perturbations in space and time

    # Amplitude of the profile position perturbations in the zonal (deg), meridional (deg), and temporal (hours) directions:
    rx = 0.5
    ry = 0.5
    rt = 0

    #
    lonc, latc = P.location.longitude, P.location.latitude
    # box = [lonc - rx / 2, lonc + rx / 2, latc - ry / 2, latc + ry / 2]

    a, b = lonc - rx / 2, lonc + rx / 2
    lon = (b - a) * np.random.random_sample((nfloats,)) + a

    a, b = latc - ry / 2, latc + ry / 2
    lat = (b - a) * np.random.random_sample((nfloats,)) + a

    a, b = 0, rt
    dtim = (b - a) * np.random.random_sample((nfloats,)) + a
    dtim = np.round(dtim).astype(int)
    tim = pd.to_datetime([P.location.time + np.timedelta64(dt, 'h') for dt in dtim])
    # dtim = (b-a) * np.random.random_sample((nfloats, )) + a
    # dtim = np.round(dtim).astype(int)
    # tim2 = pd.to_datetime([this_date - np.timedelta64(dt, 'h') for dt in dtim])
    # tim = np.sort(np.concatenate([tim2, tim1]))

    # Round time to the o(5mins), same as step=timedelta(minutes=5) in the simulation params
    tim = tim.round(freq='5min')

    #
    df = pd.DataFrame(
        [tim, lat, lon, np.arange(0, nfloats) + 9000000, np.full_like(lon, 0), ['VF' for l in lon], ['?' for l in lon]],
        index=['date', 'latitude', 'longitude', 'wmo', 'cycle_number', 'institution_code', 'file']).T
    df['date'] = pd.to_datetime(df['date'])

    return df
