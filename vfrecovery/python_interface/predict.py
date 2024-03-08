import json
from vfrecovery.core_functions.predict import predict_function


def predict(
        wmo: int,
        cyc: int,
        n_predictions,
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
    results_json = predict_function(
        wmo, cyc,
        n_predictions=n_predictions,
    )
    results = json.loads(results_json)
    return results
