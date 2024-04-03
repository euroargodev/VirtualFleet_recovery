"""
The primary goal of this module is to make it easy to determine if one simulation has already been done or not

The goal is to provide a "database"-like set of functions to manage all simulations being performed

We should provide methods:
 - to record a new simulation
 - to list all past simulations
 - to search in all past simulations


What is defining a unique VFrevovery simulation ?
- WMO & CYC & n_predictions targets > 3 params (int, int, int)
- Floats configuration > at least 6 numeric parameters
- Velocity field: name, date of download, domain size > 3 params (str, datetime, int)
- Output path > 1 param (str/Path)
- Swarm size > 1 param (int)

Rq: the velocity field time frame is set by the WMO/CYC/n_predictions targets. so there's no need for it.

This first implementation relies on a simple local pickle file with a panda dataframe

"""
from typing import List, Dict
from virtualargofleet import FloatConfiguration
from pathlib import Path
import pandas as pd
import numpy as np
import warnings


from .utils import make_hash_sha256


num2str = lambda x: "%s%s" % ("%0.4d" % x, "%0.3d" % (1e3 * np.round(x, 3) - 1e3 * np.round(x)))
str2num = lambda x: int(x[0:4]) + int(x[4:]) / 1e3


class Row2Path:
    """
    >>> Row2Path(row).wmo
    >>> Row2Path(row).cyc
    >>> Row2Path(row).velocity
    >>> Row2Path(row).run
    >>> Row2Path(row).path
    """

    def __init__(self, row):
        self.row = row

    def __repr__(self):
        summary = ["<Row2Path>"]
        summary.append("%10s: %s" % ("wmo", self.wmo))
        summary.append("%10s: %s" % ("cyc", self.cyc))
        summary.append("%10s: %s" % ("velocity", self.velocity))
        summary.append("%10s: %s" % ("run", self.run))
        return "\n".join(summary)

    @property
    def wmo(self):
        return Path(str(self.row['wmo']))

    @property
    def cyc(self):
        # |----CYC
        last = str(self.row['cyc'])
        return self.wmo.joinpath(Path(last))

    @property
    def velocity(self):
        #      |----VELOCITY(NAME + DOWNLOAD_DATE + DOMAIN_SIZE)
        last = "%s_%s_%s" % (self.row['velocity_name'],
                             num2str(self.row['velocity_domain_size']),
                             self.row['velocity_download'].strftime("%Y%m%d"),
                             )
        return self.cyc.joinpath(Path(last))

    @property
    def run(self):
        #           |---- RUN_PARAMS(NP + CFG + NF)
        last = "%s_%s_%s" % ("NP%0.3d" % self.row['n_predictions'],
                             "SW%0.4d" % self.row['swarm_size'],
                             "CFG%s" % "c".join(
                                 [num2str(self.row[c]) for c in self.row.index if "cfg_" in c]),
                             )
        return self.velocity.joinpath(Path(last))

    @property
    def path(self):
        return self.run


class Path2Row:
    """
    >>> Path2Row(path).wmo
    >>> Path2Row(path).cyc
    >>> Path2Row(path).velocity
    >>> Path2Row(path).run
    >>> Path2Row(path).row
    """

    cfg_cols: list = ['cfg_cycle_duration',
                      'cfg_life_expectancy',
                      'cfg_parking_depth',
                      'cfg_profile_depth',
                      'cfg_reco_free_surface_drift',
                      'cfg_vertical_speed']

    def __init__(self, p):
        self.path = p

    def __repr__(self):
        summary = ["<Path2Row>"]
        summary.append("%10s: %s" % ("wmo", self.wmo))
        summary.append("%10s: %s" % ("cyc", self.cyc))
        summary.append("%10s: %s" % ("velocity", self.velocity))
        summary.append("%10s: %s" % ("run", self.run))
        return "\n".join(summary)

    @property
    def wmo(self):
        return int(self.path.parts[0])

    @property
    def cyc(self):
        return int(self.path.parts[1])

    @property
    def velocity(self):
        velocity = self.path.parts[2]
        result = {}
        result.update({'velocity_name': velocity.split("_")[0]})
        result.update({'velocity_domain_size': str2num(velocity.split("_")[1])})
        result.update({'velocity_download': pd.to_datetime(velocity.split("_")[2], utc=True)})
        return result

    @property
    def run(self):
        run = self.path.parts[3]
        result = {}
        result.update({'n_predictions': int(run.split("_")[0][2:])})
        result.update({'swarm_size': int(run.split("_")[1][2:])})
        for key, value in zip(self.cfg_cols, [str2num(v) for v in run.split("_")[2][3:].split("c")]):
            result.update({key: value})
        return result

    @property
    def row(self):
        row = {}
        row.update({'wmo': self.wmo})
        row.update({'cyc': self.cyc})
        row.update({'n_predictions': int(self.run['n_predictions'])})
        for key in self.cfg_cols:
            row.update({key: self.run[key]})
        for key in self.velocity:
            row.update({key: self.velocity[key]})
        row.update({'swarm_size': int(self.run['swarm_size'])})
        return pd.DataFrame([row])


class DB:
    """

    >>> DB.dbfile
    >>> DB.init()
    >>> DB.clear()
    >>> DB.isconnected()
    >>> DB.read_data()  # Return db content as :class:`pd.DataFrame`

    >>> data = {'wmo': 6903091, 'cyc': 120, 'n_predictions': 0, 'cfg': FloatConfiguration('recovery'), 'velocity': {'name': 'GLORYS', 'download': pd.to_datetime('now', utc=True), 'domain_size': 5}, 'path_root': Path('.'), 'swarm_size': 1000}
    >>> DB.from_dict(data).checkin()  # save to db
    >>> DB.from_dict(data).checkout()  # delete from db
    >>> DB.from_dict(data).checked
    >>> DB.from_dict(data).uid
    >>> DB.from_dict(data).record

    >>> partial_data = {'wmo': 6903091}
    >>> DB.from_dict(partial_data)  # Create new instance for actions
    >>> DB.from_dict(partial_data).record

    """
    wmo: int
    cyc: int
    n_predictions: int
    cfg: FloatConfiguration
    velocity: Dict
    swarm_size: int
    path_root: Path

    required: List = ['wmo', 'cyc', 'n_predictions', 'cfg_cycle_duration',
                      'cfg_life_expectancy', 'cfg_parking_depth', 'cfg_profile_depth',
                      'cfg_reco_free_surface_drift', 'cfg_vertical_speed', 'velocity_name',
                      'velocity_download', 'velocity_domain_size', 'swarm_size', 'path_root']
    properties: List = ['wmo', 'cyc', 'n_predictions', 'cfg', 'velocity', 'swarm_size', 'path_root']

    _data: pd.DataFrame
    dbfile: Path = (Path(__file__).parent.parent).joinpath('static').joinpath('assets').joinpath(
        "simulations_registry.pkl")

    def __init__(self, **kwargs):
        # for key in self.required:
        #     if key not in kwargs:
        #         raise ValueError("Missing '%s' property" % key)
        for key in kwargs:
            if key in self.properties:
                setattr(self, key, kwargs[key])

        # Connect to database:
        self.connect()

    @classmethod
    def isconnected(cls) -> bool:
        return cls.dbfile.exists()

    @classmethod
    def clear(cls):
        def confirm():
            """
            Ask user to enter Y or N (case-insensitive).
            :return: True if the answer is Y.
            :rtype: bool
            """
            answer = ""
            while answer not in ["y", "n"]:
                answer = input("Confirm to permanently clear the simulations registry [Y/N]? ").lower()
            return answer == "y"

        if confirm():
            return cls.dbfile.unlink(missing_ok=True)

    @classmethod
    def init(cls):
        df = pd.DataFrame({'wmo': pd.Series(dtype='int'),
                           'cyc': pd.Series(dtype='int'),
                           'n_predictions': pd.Series(dtype='int'),
                           'cfg_cycle_duration': pd.Series(dtype='float'),
                           'cfg_life_expectancy': pd.Series(dtype='float'),
                           'cfg_parking_depth': pd.Series(dtype='float'),
                           'cfg_profile_depth': pd.Series(dtype='float'),
                           'cfg_reco_free_surface_drift': pd.Series(dtype='float'),
                           'cfg_vertical_speed': pd.Series(dtype='float'),
                           'velocity_name': pd.Series(dtype='string'),
                           'velocity_download': pd.Series(dtype='datetime64[ns]'),
                           'velocity_domain_size': pd.Series(dtype='float'),
                           'swarm_size': pd.Series(dtype='int'),
                           'path_root': pd.Series(dtype='object'),
                           'path': pd.Series(dtype='object'),
                           'uid': pd.Series(dtype='string'),
                           })
        df.name = pd.to_datetime('now', utc=True)
        cls._data = df
        return cls

    @classmethod
    def connect(cls):
        """Connect to database and refresh data holder"""
        if not cls.isconnected():
            cls.init()
            cls._data.to_pickle(cls.dbfile)
        else:
            cls._data = pd.read_pickle(cls.dbfile)

            # Add read-only columns generated on the fly:
            if cls._data.shape[0] > 0:
                cls._data['path'] = cls._data.apply(lambda row: Row2Path(row).path, axis=1)
                cls._data['uid'] = cls._data.apply(lambda row: make_hash_sha256(row.to_dict()), axis=1)
        return cls

    # @classmethod
    # def consolidate(cls):
    #     """Reconcile DB records with files on disk
    #
    #     Make sure that all records have result files on disk
    #     """
    #     def has_result_file(df_row):
    #         df_row['output']
    #
    #
    #     cls.connect()
    #     df = cls._data
    #     df.apply(has_result_file, axis=1)

    @classmethod
    def read_data(cls):
        """Return database content as a :class:`pd.DataFrame`"""
        cls.connect()
        return cls._data

    @classmethod
    def exists(cls, dict_of_values):
        df = cls.read_data()
        v = df.iloc[:, 0] == df.iloc[:, 0]
        for key, value in dict_of_values.items():
            v &= (df[key] == value)
        return v.any()

    @classmethod
    def put_data(cls, row):
        if not cls.exists(row):
            df = cls.read_data()
            if 'path' in row:
                warnings.warn("'path' is a read only property, removed from input")
                row.pop('path', None)
            if 'uid' in row:
                warnings.warn("'uid' is a read only property, removed from input")
                row.pop('uid', None)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_pickle(cls.dbfile)
        else:
            print("This record is already in the database")

    @classmethod
    def del_data(cls, row):
        df = cls.read_data()
        v = df.iloc[:, 0] == df.iloc[:, 0]
        for key, value in row.items():
            v &= (df[key] == value)
        df = df[v != True]
        df.to_pickle(cls.dbfile)

    @classmethod
    def get_data(cls, row):
        df = cls.read_data()
        mask = df.iloc[:, 0] == df.iloc[:, 0]
        for key in row:
            if row[key] is not None:
                mask &= df[key] == row[key]
        return df[mask]

    @classmethod
    def info(cls) -> str:
        return cls.__repr__(cls)

    def __repr__(self):
        self.connect()
        summary = ["<VFRecovery.DB>"]

        summary.append("db_file: %s" % self.dbfile)
        summary.append("connected: %s" % self.isconnected())
        summary.append("Number of records: %i" % self.read_data().shape[0])

        return "\n".join(summary)

    @staticmethod
    def from_dict(obj: Dict) -> "DB":
        return DB(**obj)

    def _instance2row(self):
        row = {}
        for key in ['wmo', 'cyc', 'n_predictions', 'cfg_cycle_duration',
                    'cfg_life_expectancy', 'cfg_parking_depth', 'cfg_profile_depth',
                    'cfg_reco_free_surface_drift', 'cfg_vertical_speed', 'velocity_name',
                    'velocity_download', 'velocity_domain_size', 'swarm_size', 'path_root']:
            row.update({key: getattr(self, key, None)})

        if hasattr(self, 'cfg'):
            for key in self.cfg.mission:
                row.update({"cfg_%s" % key: self.cfg.mission[key]})

        if hasattr(self, 'velocity'):
            for key in ['name', 'download', 'domain_size']:
                if key in self.velocity:
                    row.update({"velocity_%s" % key: self.velocity[key]})

        if hasattr(self, 'path_root'):
            row.update({'path_root': str(getattr(self, 'path_root', None))})

        return row

    def checkin(self):
        """Add one new record to the database"""
        new_row = self._instance2row()

        for key, value in new_row.items():
            if value is None:
                raise ValueError("Cannot checkin a new record with missing value for '%s'" % key)

        self.put_data(new_row)

    def checkout(self):
        """Remove record from the database"""
        row = self._instance2row()

        for key, value in row.items():
            if value is None:
                raise ValueError("Cannot id a record to remove with missing value for '%s'" % key)

        self.del_data(row)

    @property
    def checked(self):
        row = self._instance2row()
        return self.exists(row)

    @property
    def uid(self):
        row = self._instance2row()
        return self.get_data(row)['uid'].values[0]

    @property
    def path(self):
        row = self._instance2row()
        return self.get_data(row)['path'].values[0]

    @property
    def path_obj(self):
        row = self._instance2row()
        return Row2Path(pd.DataFrame([row]).iloc[0])

    @property
    def record(self) -> pd.DataFrame:
        row = self._instance2row()
        return self.get_data(row)
