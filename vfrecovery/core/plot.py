import logging
import xarray as xr
from virtualargofleet import Velocity

from vfrecovery.utils.misc import list_float_simulation_folders
import vfrecovery.plots.velocity as pltvel


root_logger = logging.getLogger("vfrecovery_root_logger")
plot_logger = logging.getLogger("vfrecovery_plot")


class log_this:

    def __init__(self, txt, log_level):
        """Log text to simulation and possibly root logger(s)"""
        getattr(root_logger, log_level.lower())(txt)
        getattr(plot_logger, log_level.lower())(txt)

    @staticmethod
    def info(txt) -> 'log_this':
        return log_this(txt, 'INFO')

    @staticmethod
    def debug(txt) -> 'log_this':
        return log_this(txt, 'DEBUG')

    @staticmethod
    def warning(txt) -> 'log_this':
        return log_this(txt, 'WARNING')

    @staticmethod
    def error(txt) -> 'log_this':
        return log_this(txt, 'ERROR')


def plot_velocity(
        wmo: int,
        cyc: int,
        log_level: str,
):
    if log_level == "QUIET":
        root_logger.disabled = True
        log_level = "CRITICAL"
    root_logger.setLevel(level=getattr(logging, log_level.upper()))

    # List folders to examine:
    plist = list_float_simulation_folders(wmo, cyc)

    #
    for c in plist.keys():
        p = plist[c]
        log_this.info("Velocity figure(s) for WMO=%s / CYC=%s:" % (wmo, c))
        ilist = sorted(p.glob("velocity_*.png"))
        if len(ilist) > 0:
            [log_this.info("\t- %s" % i) for i in ilist]
        else:
            log_this.info("No velocity figures ! Generating new ones from velocity files")

            # Load velocity field
            vlist = sorted(p.glob("velocity_*.nc"))
            for v in vlist:
                log_this.info("Loading '%s'" % v)
                # ds_vel = xr.open_dataset(v)
                # VEL = Velocity(model='GLORYS12V1' if 'GLORYS' in str(v) else 'ARMOR3D', src=ds_vel)
                # pltvel.plot(VEL, wmo, cyc, save_figure=False, workdir=p)
