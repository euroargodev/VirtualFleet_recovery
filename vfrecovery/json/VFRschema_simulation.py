from typing import List, Dict
from .VFRschema import VFvalidators
from .VFRschema_profile import Profile
from .VFRschema_meta import MetaData


class Simulation(VFvalidators):
    initial_profile: Profile
    observations: List[Profile]
    predictions: List[Profile]
    meta_data: MetaData

    schema: str = "VFrecovery-schema-simulation"
    description: str = "This document records the details of one VirtualFleet-Recovery simulation and Argo float profile predictions"
    required: List = ["initial_profile", "observations", "predictions"]
    properties: List = ["initial_profile", "observations", "predictions", "meta_data", "description"]

    @staticmethod
    def from_dict(obj: Dict) -> 'Simulation':
        return Simulation(**obj)
