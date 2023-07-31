from predictions.pc.ml import XGBoost
from predictions.pc.nn import Reservoir, NHits
from predictions.pc.stats import AutoArima, Ces, Garch


def do_method_return_extra_params(method):
    return method == AutoArima


def get_method_by_name(name: str):
    if name == 'AutoArima':
        return AutoArima
    elif name == 'Ces':
        return Ces
    elif name == 'Garch':
        return Garch
    elif name == 'XGBoost':
        return XGBoost
    elif name == 'Reservoir':
        return Reservoir
    elif name == 'NHits':
        return NHits
