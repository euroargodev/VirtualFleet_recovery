import sys
import logging

log = logging.getLogger("vfrecovery.cli")


PREF = "\033["
RESET = f"{PREF}0m"
class COLORS:
    black = "30m"
    red = "31m"
    green = "32m"
    yellow = "33m"
    blue = "34m"
    magenta = "35m"
    cyan = "36m"
    white = "37m"


def puts(text, color=None, bold=False, file=sys.stdout):
    """Alternative to print, uses no color by default but accepts any color from the COLORS class.

    Parameters
    ----------
    text
    color=None
    bold=False
    file=sys.stdout
    """
    if color is None:
        txt = f'{PREF}{1 if bold else 0}m' + text + RESET
        print(txt, file=file)
    else:
        txt = f'{PREF}{1 if bold else 0};{color}' + text + RESET
        print(txt, file=file)
    log.info(text)

