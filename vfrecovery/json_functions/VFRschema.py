"""

Re-usable base class

"""

import json
import numpy as np
import pandas as pd
import ipaddress
from typing import List, Dict, Union
import jsonschema
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from pathlib import Path
import logging

log = logging.getLogger("vfrecovery.json.schema")


class VFschema:
    """A base class to export json files following a schema"""
    schema_root: str = "https://raw.githubusercontent.com/euroargodev/VirtualFleet_recovery/json-schema/schemas"

    def __init__(self, **kwargs):
        for key in self.required:
            if key not in kwargs:
                raise ValueError("Missing '%s' property" % key)
        for key in kwargs:
            if key in self.properties:
                setattr(self, key, kwargs[key])

    def __repr__(self):
        name = self.__class__.__name__
        summary = []
        for p in self.properties:
            if p != 'description':
                summary.append("%s=%s" % (p, getattr(self, p)))
        if hasattr(self, 'description'):
            summary.append("%s='%s'" % ('description', getattr(self, 'description')))

        return "%s(%s)" % (name, ", ".join(summary))

    def _repr_html_(self):
        # return self.__repr__()
        name = self.__class__.__name__
        html = ""
        if hasattr(self, 'description'):
            html = "<p style='margin:0px'><b>%s</b>: <i>%s</i>" % (name, getattr(self, 'description'))
        props = ["<ul>"]
        for p in self.properties:
            if p != 'description':
                props.append("<li style='margin:0px'><small>%s: %s</small></li>" % (p, getattr(self, p)))
        props.append("</ul>")
        html = "%s\n%s</p>" % (html, "".join(props))
        return html

    class JSONEncoder(json.JSONEncoder):
        """Make sure all dtype are serializable"""
        def default(self, obj):
            if isinstance(obj, pd._libs.tslibs.nattype.NaTType):
                return None
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, pd.Timedelta):
                return obj.isoformat()
            if getattr(type(obj), '__name__') in ['Location', 'Profile',
                                'Metrics', 'TrajectoryLengths', 'PairwiseDistances', 'PairwiseDistancesState',
                                'SurfaceDrift', 'Transit',
                                'MetaDataSystem', 'MetaDataComputation', 'MetaData']:
                # We use "getattr(type(obj), '__name__')" in order to avoid circular import
                return obj.__dict__
            if getattr(type(obj), '__name__') in ['FloatConfiguration', 'ConfigParam']:
                return json.loads(obj.to_json(indent=0))

            # 👇️ otherwise use the default behavior
            return json.JSONEncoder.default(self, obj)

    @property
    def __dict__(self):
        d = {}
        for key in self.properties:
            if key != "description":
                value = getattr(self, key)
                d.update({key: value})
        return d

    def to_json(self, fp=None, indent=4):
        jsdata = self.__dict__
        if hasattr(self, 'schema'):
            jsdata.update({"$schema": "%s/%s.json" % (self.schema_root, getattr(self, 'schema'))})
        if fp is None:
            return json.dumps(jsdata, indent=indent, cls=self.JSONEncoder)
        else:
            return json.dump(jsdata, fp, indent=indent, cls=self.JSONEncoder)


class VFvalidators(VFschema):

    @staticmethod
    def validate(data, schema) -> Union[bool, List]:
        # Read schema and create validator:
        schema = json.loads(Path(schema).read_text())
        res = Resource.from_contents(schema)
        registry = Registry(retrieve=res)
        validator = jsonschema.Draft202012Validator(schema, registry=registry)

        # Read data and validate against schema:
        data = json.loads(Path(data).read_text())
        # return validator.validate(data)
        try:
            validator.validate(data)
        except jsonschema.exceptions.ValidationError:
            pass
        except jsonschema.exceptions.SchemaError:
            log.debug("SchemaError")
            raise
        except jsonschema.exceptions.UnknownType:
            log.debug("UnknownType")
            raise
        except jsonschema.exceptions.UndefinedTypeCheck:
            log.debug("UndefinedTypeCheck")
            raise

        errors = list(validator.iter_errors(data))
        return True if len(errors) == 0 else errors

    def _is_numeric(self, x, name='?'):
        assert isinstance(x, (int, float)), "'%s' must be a float, got '%s'" % (name, type(x))

    def _is_datetime(self, x, name='?'):
        assert isinstance(x, (
        pd.Timestamp, pd._libs.tslibs.nattype.NaTType)), "'%s' must be castable with pd.to_datetime, got '%s'" % (
        name, type(x))

    def _is_integer(self, x, name='?'):
        assert isinstance(x, int), "'%s' must be an integer, got '%s'" % (name, type(x))

    def _is_timedelta(self, x, name='?'):
        assert isinstance(x, (pd.Timedelta)), "'%s' must be castable with pd.to_timedelta, got '%s'" % (name, type(x))

    def _validate_longitude(self, x):
        self._is_numeric(x, 'longitude')
        assert np.abs(x) <= 180, "longitude must be between -180 and 180"
        return x

    def _validate_latitude(self, x):
        self._is_numeric(x, 'latitude')
        assert np.abs(x) <= 90, "latitude must be between -90 and 90"
        return x

    def _validate_time(self, x):
        self._is_datetime(x, 'time')
        return x

    def _validate_wmo(self, x):
        if x is not None:
            self._is_integer(x, 'wmo')
            assert x > 0, "'wmo' must be positive"
        return x

    def _validate_cycle_number(self, x):
        if x is not None:
            self._is_integer(x, 'cycle_number')
            assert x > 0, "'cycle_number' must be positive"
        return x

    def _validate_ip(self, x):
        if x is not None:
            try:
                ipaddress.ip_address(x)
                return x
            except ValueError:
                raise

    def _validate_velocity(self, x):
        assert x in ['ARMOR3D', 'GLORYS'], "Velocity field must be one in ['ARMOR3D', 'GLORYS']"

