"""

    ```
    Metrics(VFvalidators)
        - trajectory_lengths: ArrayMetric
        - pairwise_distances: PairwiseDistances
        - surface_drift: SurfaceDrift
        - trajectory_lengths: TrajectoryLengths
        - transit: Transit
    ```

    ```
    PairwiseDistances(VFvalidators)
        - final_state: PairwiseDistancesState
        - initial_state: PairwiseDistancesState
        - relative_state: PairwiseDistancesState
        - overlapping
        - score
        - staggering
        - std_ratio
    ```

    ```
    SurfaceDrift(VFvalidators)
        - surface_currents_speed
        - surface_currents_speed_unit
        - unit
        - value
    ```

    ```
    TrajectoryLengths(ArrayMetric)
        - median
        - std

    PairwiseDistancesState(ArrayMetric)
        - median
        - std
        - nPDFpeaks
    ```

    ```
    Transit(VFvalidators)
        - value
        - unit
    ```

"""
from typing import List, Dict
from .VFRschema import VFvalidators


class ArrayMetric(VFvalidators):
    median: float = None
    std: float = None
    nPDFpeaks: int = None

    description: str = "Some statistics from a numerical array"
    properties: List = ["median", "std", "nPDFpeaks", "description"]
    required: List = ["median", "std"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_numeric(self.median, 'median')
        self._is_numeric(self.std, 'std')
        if self.nPDFpeaks is not None:
            self._is_integer(self.nPDFpeaks, 'nPDFpeaks')

    @staticmethod
    def from_dict(obj: Dict) -> 'ArrayMetric':
        return ArrayMetric(**obj)


class TrajectoryLengths(ArrayMetric):
    description: str = "Statistics about swarm trajectory lengths"
    properties: List = ["median", "std", "description"]
    required: List = ["median", "std"]

    @staticmethod
    def from_dict(obj: Dict) -> 'TrajectoryLengths':
        return TrajectoryLengths(**obj)


class PairwiseDistancesState(ArrayMetric):
    description: str = "Statistics about swarm profiles pairwise distances at a given time slice"
    properties: List = ["median", "std", "nPDFpeaks", "description"]
    required: List = ["median", "std", "nPDFpeaks"]

    @staticmethod
    def from_dict(obj: Dict) -> 'PairwiseDistancesState':
        return PairwiseDistancesState(**obj)


class PairwiseDistances(VFvalidators):
    final_state: PairwiseDistancesState = None
    initial_state: PairwiseDistancesState = None
    relative_state: PairwiseDistancesState = None
    overlapping: float = None
    score: float = None
    staggering: float = None
    std_ratio: float = None

    description: str = "Statistics about swarm profiles pairwise distances"
    properties: List = ["final_state", "initial_state", "relative_state", "overlapping", "score", "staggering",
                        "std_ratio", "description"]
    required: List = []

    @staticmethod
    def from_dict(obj: Dict) -> 'PairwiseDistances':
        return PairwiseDistances(**obj)


class SurfaceDrift(VFvalidators):
    surface_currents_speed: float = None
    surface_currents_speed_unit: str = "m/s"
    unit: str = 'km'
    value: float = None

    description: str = "Drift by surface currents due to the float ascent time error (difference between simulated profile time and the observed one)"
    properties: List = ["surface_currents_speed", "surface_currents_speed_unit", "unit", "value", "description"]
    required: List = ["surface_currents_speed", "value"]

    @staticmethod
    def from_dict(obj: Dict) -> 'SurfaceDrift':
        return SurfaceDrift(**obj)


class Transit(VFvalidators):
    value: float = None
    unit: str = "hour"

    description: str = "Transit time to cover the distance error (assume a 12 kts boat speed with 1 kt = 1.852 km/h)"
    properties: List = ["value", "unit", "description"]
    required: List = ["value"]

    @staticmethod
    def from_dict(obj: Dict) -> 'Transit':
        return Transit(**obj)


class Metrics(VFvalidators):
    trajectory_lengths: TrajectoryLengths = None
    pairwise_distances: PairwiseDistances = None
    surface_drift: SurfaceDrift = None
    transit: Transit = None

    schema: str = "VFrecovery-schema-metrics"
    description: str = "A set of metrics to describe/interpret one predicted VFrecovery profile location"
    properties: List = ["trajectory_lengths", "pairwise_distances", "surface_drift", "trajectory_lengths", "transit", "description"]
    required: List = []

    @staticmethod
    def from_dict(obj: Dict) -> 'Metrics':
        return Metrics(**obj)
