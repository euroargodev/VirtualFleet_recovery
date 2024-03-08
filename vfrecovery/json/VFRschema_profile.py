import pandas as pd
from typing import List, Dict
from VFRschema import VFvalidators
from VFRschema_metrics import Metrics


class Location(VFvalidators):
    longitude: float
    latitude: float
    time: pd.Timestamp

    schema: str = "VFrecovery-schema-location"
    description: str = "A set of longitude/latitude/time coordinates on Earth"
    properties: List = ["longitude", "latitude", "time", "description"]
    required: List = ["longitude", "latitude"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'time' not in kwargs:
            # setattr(self, 'time', pd.to_datetime('now', utc=True))
            setattr(self, 'time', pd.NaT)

        self._validate_longitude(self.longitude)
        self._validate_latitude(self.latitude)
        self._validate_time(self.time)

    @staticmethod
    def from_dict(obj: Dict) -> 'Location':
        return Location(**obj)


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

    @staticmethod
    def from_dict(obj: Dict) -> 'Profile':
        return Profile(**obj)
