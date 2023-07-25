from metrics.completeness import BlakeCompletenessMetric
from metrics.correctness import HeinrichCorrectnessMetric
from metrics.timeliness import HeinrichTimelinessMetric
from metrics.utils import present_relation_results
from run.configuration import company_names, create_stock, columns
from timeseries.enums import DeviationScale, SeriesColumn, DeviationRange, DeviationSource

alpha = {SeriesColumn.OPEN: 0.5,
         SeriesColumn.CLOSE: 0.5,
         SeriesColumn.ADJ_CLOSE: 0.5,
         SeriesColumn.HIGH: 0.5,
         SeriesColumn.LOW: 0.5,
         SeriesColumn.VOLUME: 1.0}
declines = {SeriesColumn.OPEN: 0.4,
            SeriesColumn.CLOSE: 0.4,
            SeriesColumn.ADJ_CLOSE: 0.3,
            SeriesColumn.HIGH: 0.4,
            SeriesColumn.LOW: 0.4,
            SeriesColumn.VOLUME: 0.7}
measurement_times = {DeviationScale.SLIGHTLY: 5,
                     DeviationScale.MODERATELY: 200,
                     DeviationScale.HIGHLY: 1000}

for company_name in company_names:
    stock = create_stock(company_name)

    metrics = [
        HeinrichCorrectnessMetric(stock, alpha),
        BlakeCompletenessMetric(stock),
        HeinrichTimelinessMetric(stock, declines, measurement_times)]

    for metric in metrics:
        deviation_range = DeviationRange.ALL if metric.get_deviation_name() != DeviationSource.TIMELINESS else DeviationRange.PARTIAL
        qualities_close = metric.relation_qualities(deviation_range, columns=columns)
        present_relation_results(metric.get_deviation_name(), company_name, qualities_close,
                                 columns=[SeriesColumn.CLOSE], verbose=False)

        qualities_all = metric.relation_qualities(DeviationRange.ALL)
        qualities_partial = metric.relation_qualities(DeviationRange.PARTIAL)
        present_relation_results(metric.get_deviation_name(), company_name, qualities_all, qualities_partial,
                                 verbose=False)

print("DONE")
