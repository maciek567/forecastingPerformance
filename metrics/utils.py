import os
from enum import Enum

from pandas import DataFrame, concat

from inout.intermediate import IntermediateProvider
from inout.paths import metrics_scores_path
from timeseries.enums import DeviationRange, DeviationSource, DeviationScale, Mitigation, mitigation_short


class MetricLevel(Enum):
    VALUES = "values"
    TUPLES = "tuples"
    RELATION = "relation"


deviations_range_label = "Range"
deviations_scale_label = "Scale"
mitigation_label = "Improve"
metric_score_label = "Score"


def print_relation_results(source: DeviationSource, company_name: str, qualities_all: dict,
                           qualities_partial: dict) -> None:
    results = DataFrame(
        columns=[deviations_range_label, deviations_scale_label, mitigation_label, metric_score_label])
    for deviation_range in DeviationRange:
        qualities = qualities_partial if deviation_range == DeviationRange.PARTIAL else qualities_all
        for scale in DeviationScale:
            mitigations = Mitigation if source != DeviationSource.TIMELINESS else [Mitigation.NOT_MITIGATED]
            for mitigation in mitigations:
                result = {deviations_range_label: deviation_range.value,
                          deviations_scale_label: scale.value,
                          mitigation_label: mitigation_short()[mitigation],
                          metric_score_label: qualities[scale][mitigation]}
                results = concat([results, DataFrame([result])], ignore_index=True)

    print(f"Relation quality differences for {source.value} series:")
    print(results)
    print()
    print_relation_results_to_latex(results, source, company_name)


def print_relation_results_to_latex(results, source: DeviationSource, company_name: str) -> None:
    latex = results.to_latex(index=False,
                             formatters={"name": str.upper},
                             float_format="{:.3f}".format)
    print(latex)
    name = f"{company_name}_{source.name}"
    IntermediateProvider.save_as_latex(os.path.join(metrics_scores_path, name), latex)
