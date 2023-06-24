import os
import sys

sys.path.append('..')
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

from predictions.hpc.arimaSpark import AutoArimaSpark
from predictions.model import PredictionModel
from timeseries.timeseries import StockMarketSeries
from timeseries.enums import SeriesColumn, DeviationSource, DeviationScale

company_name = "Facebook"
column = SeriesColumn.CLOSE
time_series_start = "2017-01-03"
time_series_values = 300

prediction_start = 280
iterations = 2
methods = [AutoArimaSpark]

stock = StockMarketSeries(company_name, time_series_start, time_series_values,
                          weights={SeriesColumn.OPEN: 0.2,
                                   SeriesColumn.CLOSE: 0.2,
                                   SeriesColumn.ADJ_CLOSE: 0.25,
                                   SeriesColumn.HIGH: 0.15,
                                   SeriesColumn.LOW: 0.15,
                                   SeriesColumn.VOLUME: 0.05},
                          )

base_model = PredictionModel(stock, prediction_start, column, iterations=iterations)

for method in methods:
    model = base_model.configure_model(method, optimize=False)
    model.plot_prediction(source=DeviationSource.NONE, save_file=True)
    # model.plot_prediction(source=DeviationSource.NOISE, scale=DeviationScale.SLIGHTLY, mitigation=True, save_file=True)
    model.compute_statistics_set(save_file=True)
