import pandas as pd
import numpy as np
from typing import List, Dict, Iterable
import argopy.plot as argoplot

from .VFRschema import VFvalidators
from .VFRschema_metrics import Metrics


class Location(VFvalidators):
    longitude: float
    latitude: float
    time: pd.Timestamp

    schema: str = "VFrecovery-schema-location"
    description: str = "A set of longitude/latitude/time coordinates on Earth"
    properties: List = ["longitude", "latitude", "time", "description"]
    required: List = ["longitude", "latitude"]

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        longitude: float
        latitude: float
        time: pd.Timestamp
        """
        super().__init__(**kwargs)
        if 'time' not in kwargs:
            # setattr(self, 'time', pd.to_datetime('now', utc=True))
            setattr(self, 'time', pd.NaT)

        self._validate_longitude(self.longitude)
        self._validate_latitude(self.latitude)
        self._validate_time(self.time)

        self.longitude = np.round(self.longitude, 3)
        self.latitude = np.round(self.latitude, 3)


    @staticmethod
    def from_dict(obj: Dict) -> 'Location':
        return Location(**obj)

    @staticmethod
    def from_tuple(obj: tuple) -> 'Location':
        if len(obj) == 2:
            d = {'longitude': obj[0], 'latitude': obj[1]}
        elif len(obj) == 3:
            d = {'longitude': obj[0], 'latitude': obj[1], 'time': obj[2]}
        elif len(obj) == 4:
            d = {'longitude': obj[0], 'latitude': obj[1], 'time': obj[2], 'description': obj[3]}
        return Location(**d)


class Profile(VFvalidators):
    location: Location
    cycle_number: int = None
    wmo: int = None
    url_float: str = None
    url_profile: str = None
    virtual_cycle_number: int = None
    metrics: Metrics = None

    schema: str = "VFrecovery-schema-profile"
    description: str = "A set of meta-data and longitude/latitude/time coordinates on Earth, for an Argo float vertical profile location"
    required: List = ["location"]
    properties: List = ["location", "cycle_number", "wmo", "url_float", "url_profile", "virtual_cycle_number", "metrics", "description"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_wmo(self.wmo)
        self._validate_cycle_number(self.cycle_number)
        self._validate_cycle_number(self.virtual_cycle_number)
        if isinstance(kwargs['location'], dict):
            self.location = Location.from_dict(kwargs['location'])

    @staticmethod
    def from_dict(obj: Dict) -> 'Profile':
        """

        Parameters
        ----------
        location: Location
        wmo: int
        cycle_number: int
        url_float: str
        url_profile: str
        virtual_cycle_number: int
        metrics: Metrics

        """
        return Profile(**obj)

    @staticmethod
    def from_ArgoIndex(df: pd.DataFrame) -> List['Profile']:
        Plist = []
        df = df.sort_values(by='date')
        for irow, this_obs in df.iterrows():
            p = Profile.from_dict({
                'location': Location.from_dict({'longitude': this_obs['longitude'],
                                                'latitude': this_obs['latitude'],
                                                'time': this_obs['date'].tz_localize('UTC')
                                                }),
                'wmo': this_obs['wmo'],
                'cycle_number': this_obs['cyc'],
                'url_float': argoplot.dashboard(wmo=this_obs['wmo'], url_only=True),
                'url_profile': argoplot.dashboard(wmo=this_obs['wmo'], cyc=this_obs['cyc'], url_only=True),
            })
            Plist.append(p)
        return Plist


class Trajectory(VFvalidators):
    locations: List[Location]

    schema: str = "VFrecovery-schema-trajectory"
    description: str = "Represents two or more VirtualFleet-Recovery locations that share a relationship"
    required: List = ["locations"]
    properties: List = ["locations", "description"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if len(kwargs['locations']) < 2:
            raise ValueError("'locations' must a be list with at least 2 elements")
        L = []
        for location in kwargs['locations']:
            if isinstance(location, dict):
                loc = Location.from_dict(location)
            elif isinstance(location, Iterable):
                loc = Location.from_tuple(location)
            else:
                raise ValueError("'locations' item must be a dictionary or an Iterable")
            L.append(loc)
        self.locations = L

    @staticmethod
    def from_dict(obj: Dict) -> 'Trajectory':
        """

        Parameters
        ----------
        locations: List[Location]
        """
        return Trajectory(**obj)

    @staticmethod
    def from_tuple(obj: tuple) -> 'Trajectory':
        return Trajectory(**{'locations': obj})
