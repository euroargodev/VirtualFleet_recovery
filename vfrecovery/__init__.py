import json
import logging.config
import time
import pathlib
from importlib.metadata import version


__version__ = version("vfrecovery")

log_configuration_dict = json.load(
    open(
        pathlib.Path(
            pathlib.Path(__file__).parent, "logging_conf.json"
        )
    )
)
logging.config.dictConfig(log_configuration_dict)
logging.Formatter.converter = time.gmtime

from vfrecovery.python_interface.predict import predict
