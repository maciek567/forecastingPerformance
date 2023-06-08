import sys

sys.path.append('..')
from timeseries.timeseries import StockMarketSeries
from timeseries.utils import SeriesColumn
from predictions.model import PredictionModel
from ml import XGBoost

company_name = "Intel"
time_series_start = "2017-01-03"
time_series_values = 300
column = SeriesColumn.CLOSE
stock = StockMarketSeries(company_name, time_series_start, time_series_values,
                          weights={SeriesColumn.OPEN: 0.2,
                                   SeriesColumn.CLOSE: 0.2,
                                   SeriesColumn.ADJ_CLOSE: 0.25,
                                   SeriesColumn.HIGH: 0.15,
                                   SeriesColumn.LOW: 0.15,
                                   SeriesColumn.VOLUME: 0.05})

prediction_start = 260
iterations = 5

model = PredictionModel(stock, prediction_start, column, iterations=iterations)

xgboost = model.configure_model(XGBoost, optimize=False)

xgboost.compute_statistics_set(save_to_file=True)
