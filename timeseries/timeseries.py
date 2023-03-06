import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import Series

from metrics.utils import Strength
from metrics.utils import Incomplete
from timeseries.utils import SeriesColumn


class StockMarketSeries:
    def __init__(self):
        self.company_name = ""
        self.path = ""
        self.time_series_start = None
        self.time_series_end = None
        self.data = None
        self.series = None
        self.noises_strength = {Strength.WEAK: 0.4, Strength.MODERATE: 1.0, Strength.STRONG: 3.0}
        self.series_noised = None
        self.incomplete_parts = {Incomplete.SLIGHTLY: 0.05, Incomplete.MODERATELY: 0.12, Incomplete.HIGHLY: 0.3}
        self.series_incomplete = None

    def prepare_time_series(self, company_name: str, path: str, time_series_start: int, time_series_end: int,
                            noises_strength: dict = None, incomplete_parts: dict = None):
        self.company_name = company_name
        self.path = path
        self.time_series_start = time_series_start
        self.time_series_end = time_series_end
        self.data = pd.read_csv(self.path)
        self.series = self.create_multiple_series()
        if noises_strength is not None:
            self.noises_strength = noises_strength
        self.series_noised = {
            Strength.WEAK: self.create_multiple_series_noised(self.noises_strength[Strength.WEAK]),
            Strength.MODERATE: self.create_multiple_series_noised(self.noises_strength[Strength.MODERATE]),
            Strength.STRONG: self.create_multiple_series_noised(self.noises_strength[Strength.STRONG])}
        if incomplete_parts is not None:
            self.incomplete_parts = incomplete_parts
        self.series_incomplete = {
            Incomplete.SLIGHTLY: self.create_multiple_series_incomplete(self.incomplete_parts[Incomplete.SLIGHTLY]),
            Incomplete.MODERATELY: self.create_multiple_series_incomplete(self.incomplete_parts[Incomplete.MODERATELY]),
            Incomplete.HIGHLY: self.create_multiple_series_incomplete(self.incomplete_parts[Incomplete.HIGHLY])}

    def create_single_series(self, column_name: SeriesColumn):
        series = pd.Series(list(self.data[column_name]), index=self.data["date"])
        return series[self.time_series_start:self.time_series_end]

    def create_multiple_series(self):
        return {SeriesColumn.OPEN: self.create_single_series(SeriesColumn.OPEN.value),
                SeriesColumn.CLOSE: self.create_single_series(SeriesColumn.CLOSE.value),
                SeriesColumn.ADJ_CLOSE: self.create_single_series(SeriesColumn.ADJ_CLOSE.value),
                SeriesColumn.HIGH: self.create_single_series(SeriesColumn.HIGH.value),
                SeriesColumn.LOW: self.create_single_series(SeriesColumn.LOW.value),
                SeriesColumn.VOLUME: self.create_single_series(SeriesColumn.VOLUME.value)}

    def add_noise(self, data: Series, power: float):
        mean = 0
        std_dev = power
        noise = np.random.normal(mean, std_dev, self.time_series_end - self.time_series_start)
        return data + noise

    def add_incompleteness(self, data: Series, incomplete_part: float):
        incompleteness = np.random.choice([0, 1], self.time_series_end - self.time_series_start,
                                          p=[incomplete_part, 1.0 - incomplete_part])
        incomplete_data = []
        for i in range(self.time_series_start, self.time_series_end):
            if incompleteness[i] == 1:
                incomplete_data.append(data[i])
            else:
                incomplete_data.append(0.0)
        return incomplete_data

    def create_multiple_series_noised(self, power: float):
        return self.create_multiple_series_defected(self.add_noise, power)

    def create_multiple_series_incomplete(self, incomplete_part: float):
        return self.create_multiple_series_defected(self.add_incompleteness, incomplete_part)

    def create_multiple_series_defected(self, defect_method, defect_size: float):
        return {SeriesColumn.OPEN: defect_method(self.series[SeriesColumn.OPEN], defect_size),
                SeriesColumn.CLOSE: defect_method(self.series[SeriesColumn.CLOSE], defect_size),
                SeriesColumn.ADJ_CLOSE: defect_method(self.series[SeriesColumn.ADJ_CLOSE], defect_size),
                SeriesColumn.HIGH: defect_method(self.series[SeriesColumn.HIGH], defect_size),
                SeriesColumn.LOW: defect_method(self.series[SeriesColumn.LOW], defect_size),
                SeriesColumn.VOLUME: defect_method(self.series[SeriesColumn.VOLUME], defect_size)}

    def plot_single_series(self, data: list, column: SeriesColumn, plot_type="-"):
        plt.figure(figsize=(10, 4))
        plt.plot(data, plot_type)
        plt.title(f"{self.company_name} {column.value} price")
        plt.xlabel("Dates")
        plt.ylabel("Prices")

    def plot_multiple_series(self, title: str, **kwargs):
        fig = plt.figure(facecolor="w", figsize=(10, 4))
        ax = fig.add_subplot(111, facecolor="#dddddd", axisbelow=True)
        for label, series in (kwargs.items()):
            ax.plot(series, markersize=1.5, label=label)
        ax.set_title(f"{self.company_name} {title}")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Prices")
        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.2))
        legend.get_frame().set_alpha(0.5)
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(False)
        plt.show()
