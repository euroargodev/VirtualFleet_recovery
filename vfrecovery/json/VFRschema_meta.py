import pandas as pd
from typing import List, Dict

import platform
import socket
import psutil

from .VFRschema import VFvalidators
from virtualargofleet.utilities import VFschema_configuration


class MetaDataSystem(VFvalidators):
    architecture: str = None
    hostname: str = None
    ip_address: str = None
    platform: str = None
    platform_release: str = None
    platform_version: str = None
    processor: str = None
    ram: str = None

    schema: str = "VFrecovery-schema-system"
    description: str = "A set of meta-data to describe the system the simulation was run on"
    required: List = []
    properties: List = ["description",
                        "architecture", "hostname", "ip_address",
                        "platform", "platform_release", "platform_version",
                        "processor", "ram"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_ip(self.ip_address)

    @staticmethod
    def from_dict(obj: Dict) -> 'MetaDataSystem':
        return MetaDataSystem(**obj)

    @staticmethod
    def guess_SystemInfo() -> dict:
        """Return system information as a dict"""
        try:
            info = {}
            info['platform'] = platform.system()
            info['platform-release'] = platform.release()
            info['platform-version'] = platform.version()
            info['architecture'] = platform.machine()
            info['hostname'] = socket.gethostname()
            info['ip-address'] = socket.gethostbyname(socket.gethostname())
            # info['mac-address']=':'.join(re.findall('..', '%012x' % uuid.getnode()))
            info['processor'] = platform.processor()
            info['ram'] = str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"
            return info
        except Exception as e:
            pass

    @staticmethod
    def auto_load() -> 'MetaDataSystem':
        return MetaDataSystem(**MetaDataSystem.guess_SystemInfo())


class MetaDataComputation(VFvalidators):
    date: pd.Timestamp = None
    cpu_time: pd.Timedelta = None
    wall_time: pd.Timedelta = None

    schema: str = "VFrecovery-schema-computation"
    description: str = "A set of meta-data to describe one computation run"
    required: List = []
    properties: List = ["description",
                        "cpu_time", "wall_time", "date"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'cpu_time' not in kwargs:
            setattr(self, 'cpu_time', pd.NaT)
        else:
            self._is_timedelta(kwargs['cpu_time'], 'cpu_time')
        if 'wall_time' not in kwargs:
            setattr(self, 'wall_time', pd.NaT)
        else:
            self._is_timedelta(kwargs['wall_time'], 'wall_time')

    @staticmethod
    def from_dict(obj: Dict) -> 'MetaDataComputation':
        return MetaDataComputation(**obj)


class MetaData(VFvalidators):
    n_floats: int = None
    velocity_field: str = None
    vfconfig: VFschema_configuration = None
    computation: MetaDataComputation = None
    system: MetaDataSystem = None

    schema: str = "VFrecovery-schema-metadata"
    description: str = "A set of meta-data to describe one simulation"
    required: List = ["n_floats", "velocity_field", "vfconfig"]
    properties: List = ["description",
                        "n_floats", "velocity_field",
                        "vfconfig", "computation", "system"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_integer(self.n_floats)
        if 'vfconfig' not in kwargs:
            self.vfconfig = None

    @staticmethod
    def from_dict(obj: Dict) -> 'MetaData':
        return MetaData(**obj)

