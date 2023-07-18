import sys

from predictions.hpc.mlSpark import XGBoostSpark
from predictions.hpc.statsSpark import AutoArimaSpark, CesSpark
from predictions.model import PredictionModel
from predictions.normal.ml import XGBoost
from predictions.normal.nn import Reservoir, NHits
from predictions.normal.stats import AutoArima, Ces, Garch
from run.configuration import time_series_start, all_noises_scale, all_incompleteness_scale, all_obsolete_scale, \
    partially_noised_scales, partially_incomplete_scales, partially_obsolete_scales, time_series_values, weights, \
    company_names
from timeseries.enums import DeviationSource, DeviationScale, DeviationRange, SeriesColumn
from timeseries.timeseries import StockMarketSeries

prediction_start = 1500
iterations = 3
unique_ids = "--unique_ids" in sys.argv
methods = [Reservoir]
columns = [SeriesColumn.CLOSE]
sources = [DeviationSource.NOISE, DeviationSource.INCOMPLETENESS, DeviationSource.TIMELINESS]
scales = [DeviationScale.SLIGHTLY, DeviationScale.MODERATELY, DeviationScale.HIGHLY]
is_mitigation = True
deviation_range = DeviationRange.ALL

for company_name in company_names:
    stock = StockMarketSeries(company_name, time_series_start, time_series_values, weights,
                              all_noised_scale=all_noises_scale,
                              all_incomplete_scale=all_incompleteness_scale,
                              all_obsolete_scale=all_obsolete_scale,
                              partly_noised_scale=partially_noised_scales,
                              partly_incomplete_scale=partially_incomplete_scales,
                              partly_obsolete_scale=partially_obsolete_scales
                              )

    base_model = PredictionModel(stock, prediction_start, columns, iterations=iterations,
                                 deviation_sources=sources, deviation_scale=scales,
                                 is_deviation_mitigation=is_mitigation, deviation_range=deviation_range,
                                 unique_ids=unique_ids, is_save_predictions=True)
    for method in methods:
        model = base_model.configure_model(method)

        model.plot_prediction(source=DeviationSource.NONE, save_file=True)
        # model.plot_prediction(source=DeviationSource.NOISE, scale=DeviationScale.SLIGHTLY, mitigation=False, save_file=True)
        # model.plot_prediction(source=DeviationSource.NOISE, scale=DeviationScale.MODERATELY, mitigation=False, save_file=True)
        # model.plot_prediction(source=DeviationSource.NOISE, scale=DeviationScale.HIGHLY, mitigation=False, save_file=True)
        # model.plot_prediction(source=DeviationSource.INCOMPLETENESS, scale=DeviationScale.SLIGHTLY, mitigation=False, save_file=True)
        # model.plot_prediction(source=DeviationSource.INCOMPLETENESS, scale=DeviationScale.MODERATELY, mitigation=False, save_file=True)
        # model.plot_prediction(source=DeviationSource.INCOMPLETENESS, scale=DeviationScale.HIGHLY, mitigation=False, save_file=True)
        # model.plot_prediction(source=DeviationSource.TIMELINESS, scale=DeviationScale.SLIGHTLY, save_file=True)
        # model.plot_prediction(source=DeviationSource.TIMELINESS, scale=DeviationScale.MODERATELY,save_file=True)
        # model.plot_prediction(source=DeviationSource.TIMELINESS, scale=DeviationScale.HIGHLY, save_file=True)

        model.compute_statistics_set(save_file=True)

print("DONE")
