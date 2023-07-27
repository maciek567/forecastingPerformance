import gc
import time

from pandas import Series

from timeseries.enums import MitigationType


def perform_mitigation(series: Series, mitigation_method) -> dict:
    gc.disable()
    start_time = time.perf_counter_ns()

    mitigated_series = mitigation_method(series)

    elapsed_time_ms = (time.perf_counter_ns() - start_time) / 1e6
    gc.enable()
    return {MitigationType.DATA: mitigated_series, MitigationType.TIME: elapsed_time_ms}


def normalize_weights(weights):
    return {column: weight / sum([w for w in weights.values()]) for column, weight in weights.items()}


def normalize_with_columns(weights, columns):
    weights = {column: weights[column] for column in columns} if columns else weights
    return normalize_weights(weights)
