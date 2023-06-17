from enum import Enum

from pandas import DataFrame, concat

from timeseries.enums import DeviationRange, DeviationSource, DeviationScale


class MetricLevel(Enum):
    VALUES = "values"
    TUPLES = "tuples"
    RELATION = "relation"


deviations_source_label = "Deviation"
deviations_scale_label = "Scale"
deviations_range_label = "Range"
metric_score_label = "Score"


def print_relation_results(qualities: dict, source: DeviationSource, deviated_range: DeviationRange) -> None:
    deviated_columns = "all" if deviated_range == DeviationRange.ALL else "some"
    print(f"Relation quality differences due to different {source.value} levels of {deviated_columns} fields:")
    for scale in DeviationScale:
        print(f"Data {scale.value} deviated: {qualities[scale]}")
    print()
    print_relation_results_to_latex(qualities, source, deviated_range)


def print_relation_results_to_latex(qualities: dict, source: DeviationSource, deviated_range: DeviationRange) -> None:
    results = DataFrame(
        columns=[deviations_source_label, deviations_scale_label, deviations_range_label, metric_score_label])
    for scale in DeviationScale:
        result = {deviations_source_label: source.value,
                  deviations_scale_label: scale.value,
                  deviations_range_label: deviated_range.value,
                  metric_score_label: qualities[scale]}
        results = concat([results, DataFrame([result])], ignore_index=True)
    print(results.to_latex(index=False,
                           formatters={"name": str.upper},
                           float_format="{:.3f}".format))
