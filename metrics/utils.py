from enum import Enum

from timeseries.utils import DeviationRange, DeviationSource, DeviationScale


class MetricLevel(Enum):
    VALUES = "values"
    TUPLES = "tuples"
    RELATION = "relation"


def print_relation_results(qualities: dict, source: DeviationSource, deviated_range: DeviationRange) -> None:
    deviated_columns = "all" if deviated_range == DeviationRange.ALL else "some"
    print(f"Relation quality differences due to different {source.value} levels of {deviated_columns} fields:")
    for scale in DeviationScale:
        print(f"Data {scale.value} deviated: {qualities[scale]}")
