from math import radians, cos, sin, asin, sqrt
import pyproj
import numpy as np


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance (in [km]) between two points
    on the earth (specified in decimal degrees)

    see: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points

    Parameters
    ----------
    lon1
    lat1
    lon2
    lat2

    Returns
    -------
    km
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers.
    return c * r


def bearing(lon1, lat1, lon2, lat2):
    """

    Parameters
    ----------
    lon1
    lat1
    lon2
    lat2

    Returns
    -------

    """
    # from math import cos, sin, atan2, degrees
    # b = atan2(cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1), sin(lon2 - lon1) * cos(lat2))
    # b = degrees(b)
    # return b

    geodesic = pyproj.Geod(ellps='WGS84')
    fwd_azimuth, back_azimuth, distance = geodesic.inv(lon1, lat1, lon2, lat2)
    return fwd_azimuth


def fixLON(x):
    """Ensure a 0-360 longitude"""
    if x < 0:
        x = 360 + x
    return x

