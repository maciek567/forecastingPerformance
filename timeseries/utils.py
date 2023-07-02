import gc
import time

from pandas import Series

from timeseries.enums import Mitigation


def perform_mitigation(series: Series, mitigation_method) -> dict:
    gc.disable()
    start_time = time.perf_counter_ns()

    mitigated_series = mitigation_method(series)

    elapsed_time_ms = (time.perf_counter_ns() - start_time) / 1e6
    gc.enable()
    return {Mitigation.DATA: mitigated_series, Mitigation.TIME: elapsed_time_ms}
