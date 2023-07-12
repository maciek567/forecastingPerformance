import sys

from predictions.hpc.mlSpark import XGBoostSpark
from predictions.hpc.statsSpark import AutoArimaSpark, CesSpark
from predictions.model import PredictionModel
from predictions.normal.ml import XGBoost
from predictions.normal.nn import Reservoir, NHits
from predictions.normal.stats import AutoArima, Ces, Garch
from timeseries.enums import SeriesColumn, DeviationSource, DeviationScale
from timeseries.timeseries import StockMarketSeries

# company_names = ['AMD', 'Accenture', 'Acer', 'Activision', 'Adobe', 'Akamai', 'Alibaba', 'Amazon', 'Apple', 'At&t',
#                  'Autodesk', 'Canon', 'Capgemini', 'Cisco', 'Ericsson', 'Facebook', 'Google', 'HP', 'IBM', 'Intel',
#                  'Mastercard', 'Microsoft', 'Motorola', 'Nokia', 'Nvidia', 'Oracle', 'Sony', 'Tmobile']
# methods = [AutoArima, Ces, Garch, XGBoost, Reservoir, NHits, AutoArimaSpark, CesSpark, XGBoostSpark]

company_names = ["Accenture"]
columns = [SeriesColumn.CLOSE]
weights = {SeriesColumn.OPEN: 0.2,
           SeriesColumn.CLOSE: 0.2,
           SeriesColumn.ADJ_CLOSE: 0.25,
           SeriesColumn.HIGH: 0.15,
           SeriesColumn.LOW: 0.15,
           SeriesColumn.VOLUME: 0.05}
time_series_start = "2017-01-03"
time_series_values = 1575
noises_scale = {DeviationScale.SLIGHTLY: 0.7, DeviationScale.MODERATELY: 1.7, DeviationScale.HIGHLY: 4.0}
incompleteness_scale = {DeviationScale.SLIGHTLY: 0.05, DeviationScale.MODERATELY: 0.12, DeviationScale.HIGHLY: 0.3}
prediction_shifts = {DeviationScale.SLIGHTLY: 5, DeviationScale.MODERATELY: 15, DeviationScale.HIGHLY: 50}

prediction_start = 1500
iterations = 3
unique_ids = "--unique_ids" in sys.argv
methods = [AutoArima]
sources = [DeviationSource.NOISE, DeviationSource.INCOMPLETENESS, DeviationSource.TIMELINESS]
scales = [DeviationScale.SLIGHTLY, DeviationScale.MODERATELY, DeviationScale.HIGHLY]
is_mitigation = True

for company_name in company_names:
    stock = StockMarketSeries(company_name, time_series_start, time_series_values,
                              obsoleteness_scale=prediction_shifts,
                              all_incomplete_parts=incompleteness_scale,
                              all_noises_strength=noises_scale,
                              weights=weights
                              )

    base_model = PredictionModel(stock, prediction_start, columns, iterations=iterations, unique_ids=unique_ids,
                                 deviation_sources=sources, deviation_scale=scales,
                                 is_deviation_mitigation=is_mitigation)
    for method in methods:
        model = base_model.configure_model(method, optimize=False)

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
