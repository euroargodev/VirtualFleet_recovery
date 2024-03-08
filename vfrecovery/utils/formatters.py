
def strfdelta(tdelta, fmt):
    """

    Parameters
    ----------
    tdelta
    fmt

    Returns
    -------

    """
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)

