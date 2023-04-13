from enum import Enum

from timeseries.utils import DefectionRange, DefectsSource, DefectsScale


class MetricLevel(Enum):
    VALUES = "values"
    TUPLES = "tuples"
    RELATION = "relation"


def print_relation_results(qualities: dict, source: DefectsSource, defection_range: DefectionRange) -> None:
    defected_columns = "all" if defection_range == DefectionRange.ALL else "some"
    print(f"Relation quality differences due to different {source.value} levels of {defected_columns} fields:")
    for scale in DefectsScale:
        print(f"Data {scale.value} defected: {qualities[scale]}")
