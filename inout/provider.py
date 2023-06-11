from datetime import datetime

import yfinance as yf
from pandas import DataFrame, read_csv

stock_path = "../data/timeseries/stock/"


class YFinanceProvider:

    @staticmethod
    def download_data(ticker, start_date, end_date, interval) -> DataFrame:
        df = yf.download(tickers=ticker, start=start_date, end=end_date, interval=interval)

        return df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
                                  'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'})

    @staticmethod
    def save_as_csv(dataframe, name: str) -> None:
        dataframe.to_csv(YFinanceProvider.get_csv_path(name))

    @staticmethod
    def load_csv(name: str) -> DataFrame:
        return read_csv(YFinanceProvider.get_csv_path(name))

    @staticmethod
    def get_csv_path(name: str) -> str:
        return stock_path + name + ".csv"

    @staticmethod
    def get_first_dates_sorted(companies: dict) -> list:
        first_dates = {}
        for name in companies.keys():
            df = YFinanceProvider.load_csv(name)
            date = datetime.strptime(df['Date'][0], '%Y-%m-%d')
            first_dates[name] = date
        return [item for item in sorted(first_dates.items(), key=lambda val: val[1])]
