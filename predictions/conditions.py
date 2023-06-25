from predictions.hpc.arimaSpark import AutoArimaSpark
from predictions.normal.arima import AutoArimaSF


def are_method_results_undeterministic(method, spark):
    if spark is None:
        from predictions.normal.ml import Reservoir
        return method == Reservoir


def do_method_return_extra_params(method):
    return method == AutoArimaSF or method == AutoArimaSpark
