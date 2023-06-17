import sys

from shared import obsolescence_scale, company_name

sys.path.append('..')
from timeseries.timeseries import StockMarketSeries
from timeseries.enums import SeriesColumn

time_series_start = "2017-01-03"
time_series_values = 300

stock = StockMarketSeries(company_name, time_series_start, time_series_values,
                          columns={SeriesColumn.CLOSE},
                          weights={SeriesColumn.OPEN: 0.2,
                                   SeriesColumn.CLOSE: 0.2,
                                   SeriesColumn.ADJ_CLOSE: 0.25,
                                   SeriesColumn.HIGH: 0.15,
                                   SeriesColumn.LOW: 0.15,
                                   SeriesColumn.VOLUME: 0.05},
                          obsoleteness_scale=obsolescence_scale,
                          cache=True)
