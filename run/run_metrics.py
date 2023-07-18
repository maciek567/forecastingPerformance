from metrics.completeness import BlakeCompletenessMetric
from metrics.correctness import HeinrichCorrectnessMetric
from metrics.timeliness import HeinrichTimelinessMetric
from metrics.utils import print_relation_results
from run.configuration import time_series_start, all_noises_scale, all_incompleteness_scale, all_obsolete_scale, \
    partially_noised_scales, partially_incomplete_scales, partially_obsolete_scales, time_series_values, weights, \
    company_names
from timeseries.enums import DeviationScale, SeriesColumn, DeviationRange
from timeseries.timeseries import StockMarketSeries

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
    stock = StockMarketSeries(company_name, time_series_start, time_series_values, weights,
                              all_noised_scale=all_noises_scale,
                              all_incomplete_scale=all_incompleteness_scale,
                              all_obsolete_scale=all_obsolete_scale,
                              partly_noised_scale=partially_noised_scales,
                              partly_incomplete_scale=partially_incomplete_scales,
                              partly_obsolete_scale=partially_obsolete_scales
                              )

    metrics = [
        HeinrichCorrectnessMetric(stock, alpha),
        BlakeCompletenessMetric(stock),
        HeinrichTimelinessMetric(stock, declines, measurement_times)]

    for metric in metrics:
        qualities_all = metric.relation_qualities(DeviationRange.ALL)
        qualities_partial = metric.relation_qualities(DeviationRange.PARTIAL)
        print_relation_results(metric.get_deviation_name(), company_name, qualities_all, qualities_partial)

print("DONE")
