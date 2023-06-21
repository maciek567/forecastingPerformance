from predictions.normal.arima import AutoArimaSF
from predictions.normal.model import PredictionModel
from run.shared import company_name, column
from timeseries.enums import SeriesColumn, DeviationSource
from timeseries.timeseries import StockMarketSeries

time_series_start = "2017-01-03"
time_series_values = 300
stock = StockMarketSeries(company_name, time_series_start, time_series_values,
                          weights={SeriesColumn.OPEN: 0.2,
                                   SeriesColumn.CLOSE: 0.2,
                                   SeriesColumn.ADJ_CLOSE: 0.25,
                                   SeriesColumn.HIGH: 0.15,
                                   SeriesColumn.LOW: 0.15,
                                   SeriesColumn.VOLUME: 0.05})

prediction_start = 260
iterations = 2
method = AutoArimaSF

base_model = PredictionModel(stock, prediction_start, column, iterations=iterations)
model = base_model.configure_model(method, optimize=False)
# model.plot_prediction(source=DeviationSource.NONE)
model.compute_statistics_set(save_to_file=True)
