import time

from pandas import Series

from timeseries.enums import Mitigation


def perform_mitigation(series: Series, mitigation_method, multiple_runs: bool = False) -> dict:
    start_time = time.time_ns()
    mitigated_series = mitigation_method(series)
    elapsed_time_ms = 0.0

    if not multiple_runs:
        elapsed_time_ms = (time.time_ns() - start_time) / 1e6
    if multiple_runs:
        for i in range(99):
            mitigation_method(series)
        elapsed_time_ms = (time.time_ns() - start_time) / 1e6 / 100

    return {Mitigation.DATA: mitigated_series, Mitigation.TIME: elapsed_time_ms}
