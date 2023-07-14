import sys

from predictions.hpc.mlSpark import XGBoostSpark
from predictions.hpc.statsSpark import AutoArimaSpark, CesSpark
from predictions.model import PredictionModel
from predictions.normal.ml import XGBoost
from predictions.normal.nn import Reservoir, NHits
from predictions.normal.stats import AutoArima, Ces, Garch
from timeseries.enums import SeriesColumn, DeviationSource, DeviationScale, DeviationRange
from timeseries.timeseries import StockMarketSeries

# company_names = ['AMD', 'Accenture', 'Acer', 'Activision', 'Adobe', 'Akamai', 'Alibaba', 'Amazon', 'Apple', 'At&t',
#                  'Autodesk', 'Canon', 'Capgemini', 'Cisco', 'Ericsson', 'Facebook', 'Google', 'HP', 'IBM', 'Intel',
#                  'Mastercard', 'Microsoft', 'Motorola', 'Nokia', 'Nvidia', 'Oracle', 'Sony', 'Tmobile']
# methods = [AutoArima, Ces, Garch, XGBoost, Reservoir, NHits, AutoArimaSpark, CesSpark, XGBoostSpark]

company_names = ["Accenture"]
columns = [SeriesColumn.LOW, SeriesColumn.HIGH]
time_series_start = "2017-01-03"
time_series_values = 1575

weights = {SeriesColumn.OPEN: 0.2, SeriesColumn.CLOSE: 0.2, SeriesColumn.ADJ_CLOSE: 0.25,
           SeriesColumn.HIGH: 0.15, SeriesColumn.LOW: 0.15, SeriesColumn.VOLUME: 0.05}
all_noises_scale = {DeviationScale.SLIGHTLY: 0.7, DeviationScale.MODERATELY: 1.7, DeviationScale.HIGHLY: 4.0}
all_incompleteness_scale = {DeviationScale.SLIGHTLY: 0.05, DeviationScale.MODERATELY: 0.12, DeviationScale.HIGHLY: 0.3}
all_obsolete_scale = {DeviationScale.SLIGHTLY: 5, DeviationScale.MODERATELY: 15, DeviationScale.HIGHLY: 50}
partially_noised_scales = \
    {SeriesColumn.CLOSE: {DeviationScale.SLIGHTLY: 0.6, DeviationScale.MODERATELY: 2.0, DeviationScale.HIGHLY: 6.0},
     SeriesColumn.OPEN: {DeviationScale.SLIGHTLY: 0.4, DeviationScale.MODERATELY: 1.7, DeviationScale.HIGHLY: 5.2}}
partially_incomplete_scales = \
    {SeriesColumn.CLOSE: {DeviationScale.SLIGHTLY: 0.05, DeviationScale.MODERATELY: 0.12, DeviationScale.HIGHLY: 0.3},
     SeriesColumn.OPEN: {DeviationScale.SLIGHTLY: 0.03, DeviationScale.MODERATELY: 0.08, DeviationScale.HIGHLY: 0.18}}
partially_obsolete_scales = \
    {SeriesColumn.CLOSE: {DeviationScale.SLIGHTLY: 5, DeviationScale.MODERATELY: 20, DeviationScale.HIGHLY: 50},
     SeriesColumn.OPEN: {DeviationScale.SLIGHTLY: 3, DeviationScale.MODERATELY: 12, DeviationScale.HIGHLY: 30}}

prediction_start = 1500
iterations = 3
unique_ids = "--unique_ids" in sys.argv
methods = [Ces]
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
