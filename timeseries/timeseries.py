import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series

from provider.provider import YFinanceProvider
from timeseries.incompleteness import IncompleteSeries
from timeseries.noise import NoisedSeries
from timeseries.obsolescence import ObsolescenceSeries
from timeseries.utils import SeriesColumn, DeviationSource, DeviationRange, DeviationScale, save_image, set_legend


class StockMarketSeries:
    def __init__(self, company_name: str, time_series_start: str, time_series_values: int, weights: dict,
                 all_noises_strength: dict = None, all_incomplete_parts: dict = None, obsoleteness_scale: dict = None,
                 partially_noised_strength: dict = None, partially_incomplete_parts: dict = None):
        self.company_name = company_name
        self.data = YFinanceProvider.load_csv(company_name)
        self.time_series_start = self.find_start_date(time_series_start)
        self.time_series_end = self.time_series_start + time_series_values
        self.real_series = self.create_multiple_series()
        self.weights = weights
        self.all_deviated_series = {}
        self.partially_deviated_series = {}
        self.mitigated_deviations_series = {}
        self.noises = NoisedSeries(self, all_noises_strength, partially_noised_strength)
        self.incompleteness = IncompleteSeries(self, all_incomplete_parts, partially_incomplete_parts)
        self.obsolescence = ObsolescenceSeries(self, obsoleteness_scale)

    def find_start_date(self, time_series_start: str) -> int:
        return self.data.index[self.data['Date'] == time_series_start].values.tolist()[0]

    def create_single_series(self, column_name: SeriesColumn, extra_days: int) -> Series:
        series = pd.Series(list(self.data[column_name]), index=self.data["Date"])
        return series[self.time_series_start:self.time_series_end + extra_days]

    def create_multiple_series(self, extra_days: int = 0) -> dict:
        return {column: self.create_single_series(column.value, extra_days) for column in SeriesColumn}

    @staticmethod
    def get_list_for_tuple(series: dict, i: int) -> list:
        return [series[column][i] for column in SeriesColumn]

    @staticmethod
    def get_dict_for_tuple(series: dict, i: int) -> dict:
        return {column: series[column][i] for column in SeriesColumn}

    def get_deviated_series(self, source: DeviationSource,
                            deviation_range: DeviationRange = DeviationRange.ALL) -> dict:
        return self.all_deviated_series[source] if deviation_range == DeviationRange.ALL \
            else self.partially_deviated_series[source]

    def deviate_all_series(self, deviations: dict) -> dict:
        return {column: deviation.method(self.real_series[column], deviation.scale) for column, deviation in
                deviations.items()}

    def deviate_some_series(self, series_to_deviate: dict) -> dict:
        return {column: self.real_series[column] if column not in series_to_deviate.keys() else None
                for column in SeriesColumn} | \
            {column: deviation.method(self.real_series[column], deviation.scale)
             for column, deviation in series_to_deviate.items()}

    def plot_series(self, title: str, **kwargs) -> None:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111, axisbelow=True)
        for label, series in (kwargs.items()):
            ax.plot(series.values, markersize=1.0, label=label)
        title = f"{self.company_name} {title}"
        ax.set_title(title)
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Prices [USD]")
        set_legend(ax)
        save_image(plt, title)
        plt.show()
