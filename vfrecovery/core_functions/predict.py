
def predict_function(
        wmo: int,
        cyc: int,
        n_predictions: int = 1,
):
    """
    Execute VirtualFleet-Recovery predictor and return results as a JSON string

    Inputs
    ------
    wmo
    cyc
    n_predictions

    Returns
    -------
    data

    """  # noqa
    return {'wmo': wmo, 'cyc': cyc}
