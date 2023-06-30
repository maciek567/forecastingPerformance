import sys

from predictions.model import PredictionModel
from predictions.normal.arima import AutoArimaSF
from timeseries.enums import SeriesColumn, DeviationSource
from timeseries.timeseries import StockMarketSeries

# company_names = ['AMD', 'Accenture', 'Acer', 'Activision', 'Adobe', 'Akamai', 'Alibaba', 'Amazon', 'Apple', 'At&t',
#                'Autodesk', 'Canon', 'Capgemini', 'Cisco', 'Ericsson', 'Facebook', 'Google', 'HP', 'IBM', 'Intel',
#               'Mastercard', 'Microsoft', 'Motorola', 'Nokia', 'Nvidia', 'Oracle', 'Samsung', 'Sony', 'Tmobile']
company_names = ["Google"]
column = SeriesColumn.CLOSE
time_series_start = "2017-01-03"
time_series_values = 1627

prediction_start = 1587
iterations = 3
unique_ids = "--unique_ids" in sys.argv
methods = [AutoArimaSF]

for company_name in company_names:
    stock = StockMarketSeries(company_name, time_series_start, time_series_values,
                              weights={SeriesColumn.OPEN: 0.2,
                                       SeriesColumn.CLOSE: 0.2,
                                       SeriesColumn.ADJ_CLOSE: 0.25,
                                       SeriesColumn.HIGH: 0.15,
                                       SeriesColumn.LOW: 0.15,
                                       SeriesColumn.VOLUME: 0.05}
                              )

    base_model = PredictionModel(stock, prediction_start, column, iterations=iterations, unique_ids=unique_ids)
    for method in methods:
        model = base_model.configure_model(method, optimize=False)
        model.plot_prediction(source=DeviationSource.NONE, save_file=True)
        # model.plot_prediction(source=DeviationSource.NOISE, scale=DeviationScale.SLIGHTLY, mitigation=True, save_file=True)
        model.compute_statistics_set(save_file=True)

print("DONE")
