import os
from datetime import datetime

import yfinance as yf
from pandas import DataFrame, read_csv

from inout.paths import stock_path


class YFinanceProvider:

    @staticmethod
    def download_data(ticker, start_date, end_date, interval) -> DataFrame:
        df = yf.download(tickers=ticker, start=start_date, end=end_date, interval=interval)

        return df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
                                  'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'})

    @staticmethod
    def save_as_csv(dataframe, name: str) -> None:
        dataframe.to_csv(YFinanceProvider.get_stock_csv_path(name))

    @staticmethod
    def load_csv(name: str) -> DataFrame:
        return read_csv(YFinanceProvider.get_stock_csv_path(name))

    @staticmethod
    def get_stock_csv_path(name: str) -> str:
        return os.path.join(stock_path, name) + ".csv"

    @staticmethod
    def get_first_dates_sorted(companies: dict) -> list:
        first_dates = {}
        for name in companies.keys():
            df = YFinanceProvider.load_csv(name)
            date = datetime.strptime(df['Date'][0], '%Y-%m-%d')
            first_dates[name] = date
        return [item for item in sorted(first_dates.items(), key=lambda val: val[1])]


possible_start_dates = ["2017-01-03", "2017-01-04", "2017-01-05", "2017-01-06", "2017-01-09", "2017-01-10",
                        "2017-01-11", "2017-01-12", "2017-01-13", "2017-01-17", "2017-01-18", "2017-01-19",
                        "2017-01-20", "2017-01-23", "2017-01-24", "2017-01-25", "2017-01-26", "2017-01-27",
                        "2017-01-30", "2017-01-31", "2017-02-01", "2017-02-02", "2017-02-03", "2017-02-06",
                        "2017-02-07", "2017-02-08", "2017-02-09", "2017-02-10", "2017-02-13", "2017-02-14",
                        "2017-02-15", "2017-02-16", "2017-02-17", "2017-02-21", "2017-02-22", "2017-02-23",
                        "2017-02-24", "2017-02-27", "2017-02-28", "2017-03-01", "2017-03-02", "2017-03-03",
                        "2017-03-06", "2017-03-07", "2017-03-08", "2017-03-09", "2017-03-10", "2017-03-13",
                        "2017-03-14", "2017-03-15"]
