import sys

from predictions.hpc.arimaSpark import AutoArimaSpark
from predictions.hpc.statisticalSpark import CesSpark
from predictions.model import PredictionModel
from predictions.normal.arima import AutoArimaSF
from predictions.normal.ml import Reservoir, NHits, XGBoost
from predictions.normal.statistical import Ces, Garch
from timeseries.enums import SeriesColumn, DeviationSource, DeviationScale
from timeseries.timeseries import StockMarketSeries

# company_names = ['AMD', 'Accenture', 'Acer', 'Activision', 'Adobe', 'Akamai', 'Alibaba', 'Amazon', 'Apple', 'At&t',
#                'Autodesk', 'Canon', 'Capgemini', 'Cisco', 'Ericsson', 'Facebook', 'Google', 'HP', 'IBM', 'Intel',
#               'Mastercard', 'Microsoft', 'Motorola', 'Nokia', 'Nvidia', 'Oracle', 'Sony', 'Tmobile']
# methods = [AutoArimaSF, Ces, Garch, XGBoost, Reservoir, NHits, AutoArimaSpark, CesSpark]

company_names = ["Facebook"]
columns = [SeriesColumn.LOW, SeriesColumn.HIGH]
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
methods = [AutoArimaSF]
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
        model.plot_prediction(source=DeviationSource.NOISE, scale=DeviationScale.SLIGHTLY, mitigation=False, save_file=True)
        model.plot_prediction(source=DeviationSource.NOISE, scale=DeviationScale.MODERATELY, mitigation=False, save_file=True)
        model.plot_prediction(source=DeviationSource.NOISE, scale=DeviationScale.HIGHLY, mitigation=False, save_file=True)
        model.plot_prediction(source=DeviationSource.INCOMPLETENESS, scale=DeviationScale.SLIGHTLY, mitigation=False, save_file=True)
        model.plot_prediction(source=DeviationSource.INCOMPLETENESS, scale=DeviationScale.MODERATELY, mitigation=False, save_file=True)
        model.plot_prediction(source=DeviationSource.INCOMPLETENESS, scale=DeviationScale.HIGHLY, mitigation=False, save_file=True)
        model.plot_prediction(source=DeviationSource.TIMELINESS, scale=DeviationScale.SLIGHTLY, save_file=True)
        model.plot_prediction(source=DeviationSource.TIMELINESS, scale=DeviationScale.MODERATELY,save_file=True)
        model.plot_prediction(source=DeviationSource.TIMELINESS, scale=DeviationScale.HIGHLY, save_file=True)

        model.compute_statistics_set(save_file=True)

print("DONE")
