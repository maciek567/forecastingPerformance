import sys

from inout.provider import possible_start_dates
from predictions.hpc.mlSpark import XGBoostSpark
from predictions.hpc.statsSpark import AutoArimaSpark, CesSpark
from predictions.model import PredictionModel
from predictions.normal.ml import XGBoost
from predictions.normal.nn import Reservoir, NHits
from predictions.normal.stats import AutoArima, Ces, Garch
from run.configuration import company_names, create_stock
from timeseries.enums import DeviationSource, DeviationScale, DeviationRange, SeriesColumn

prediction_start = 1500
iterations = 3
methods = [Ces]
columns = [SeriesColumn.CLOSE]
deviation_range = DeviationRange.ALL

sources = [DeviationSource.NOISE, DeviationSource.INCOMPLETENESS, DeviationSource.TIMELINESS]
scales = [DeviationScale.SLIGHTLY, DeviationScale.MODERATELY, DeviationScale.HIGHLY]
is_mitigation = True
graph_start = 1400
training_shifts = 0
unique_ids = "--unique_ids" in sys.argv

for company_name in company_names:
    for shift in range(0, training_shifts + 1):
        stock = create_stock(company_name, start=possible_start_dates[shift])

        base_model = PredictionModel(stock, prediction_start, columns, graph_start, iterations=iterations,
                                     deviation_sources=sources, deviation_scale=scales,
                                     is_deviation_mitigation=is_mitigation, deviation_range=deviation_range,
                                     unique_ids=unique_ids, is_save_predictions=True, shift=shift)
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

            model.plot_group(sources=[DeviationSource.NONE, DeviationSource.NOISE, DeviationSource.INCOMPLETENESS,
                                      DeviationSource.TIMELINESS],
                             scales=[None, DeviationScale.MODERATELY, DeviationScale.MODERATELY,
                                     DeviationScale.MODERATELY],
                             mitigations=[False, False, False, False], save_file=True)

            model.compute_statistics_set(save_file=True)

print("DONE")
