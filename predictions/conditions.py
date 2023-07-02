from predictions.hpc.arimaSpark import AutoArimaSpark
from predictions.normal.arima import AutoArimaSF


def do_method_return_extra_params(method):
    return method == AutoArimaSF or method == AutoArimaSpark
