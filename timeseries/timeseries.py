from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series

from timeseries.utils import SeriesColumn, DeviationScale, DeviationSource, DeviationRange, save_image, \
    set_legend

DAYS_IN_YEAR = 365


class Deviation:
    def __init__(self, method, scale):
        self.method = method
        self.scale = scale


class StockMarketSeries:
    def __init__(self, company_name: str, path: str, time_series_start: int, time_series_end: int,
                 all_noises_strength: dict = None, all_incomplete_parts: dict = None,
                 partially_noised_strength: dict = None, partially_incomplete_parts: dict = None):
        self.company_name = company_name
        self.path = path
        self.time_series_start = time_series_start
        self.time_series_end = time_series_end
        self.data = pd.read_csv(self.path)
        self.real_series = self.create_multiple_series()
        self.all_noises_strength = all_noises_strength if all_noises_strength is not None \
            else {DeviationScale.SLIGHTLY: 0.4, DeviationScale.MODERATELY: 1.0, DeviationScale.HIGHLY: 3.0}
        self.all_incomplete_parts = all_incomplete_parts if all_incomplete_parts is not None \
            else {DeviationScale.SLIGHTLY: 0.05, DeviationScale.MODERATELY: 0.12, DeviationScale.HIGHLY: 0.3}
        self.all_deviated_series = {
            DeviationSource.NOISE:
                {strength: self.noise_all_series(self.all_noises_strength[strength]) for strength in DeviationScale},
            DeviationSource.INCOMPLETENESS:
                {strength: self.nullify_all_series(self.all_incomplete_parts[strength]) for strength in DeviationScale}}
        self.partially_deviated_series = {}
        if partially_noised_strength is not None:
            self.partially_noised_strength = partially_noised_strength
            self.partially_deviated_series[DeviationSource.NOISE] = self.noise_some_series_set(
                partially_noised_strength)
        if partially_incomplete_parts is not None:
            self.partially_incomplete_parts = partially_incomplete_parts
            self.partially_deviated_series[DeviationSource.INCOMPLETENESS] = \
                self.add_incompleteness_to_some_series_set(partially_incomplete_parts)

    def create_single_series(self, column_name: SeriesColumn) -> Series:
        series = pd.Series(list(self.data[column_name]), index=self.data["date"])
        return series[self.time_series_start:self.time_series_end]

    def create_multiple_series(self) -> dict:
        return {column: self.create_single_series(column.value) for column in SeriesColumn}

    @staticmethod
    def attributes_list(series: dict, i: int) -> list:
        return [series[column][i] for column in SeriesColumn]

    @staticmethod
    def to_date(date_string: str) -> date:
        return datetime.strptime(date_string, '%Y-%m-%d').date()

    def get_ages(self, measurement_time: int = None) -> tuple:
        dates = self.real_series[SeriesColumn.OPEN].index.tolist()
        today = date.today() if measurement_time is None else self.to_date(dates[-1]) + timedelta(days=measurement_time)
        time_diffs = [str(measurement_time + len(dates) - i) for i in range(len(dates))]
        ages = [(today - self.to_date(dates[i])).days / DAYS_IN_YEAR for i in range(len(dates))]
        return time_diffs, ages

    def add_noise(self, data: Series, power: float) -> Series:
        mean = 0
        std_dev = power
        noise = np.random.normal(mean, std_dev, self.time_series_end - self.time_series_start)
        return data + noise

    def add_incompleteness(self, data: Series, incomplete_part: float) -> Series:
        incompleteness = np.random.choice([0, 1], self.time_series_end - self.time_series_start,
                                          p=[incomplete_part, 1.0 - incomplete_part])
        incomplete_data = []
        for i in range(0, self.time_series_end - self.time_series_start):
            if incompleteness[i] == 1:
                incomplete_data.append(data[i])
            else:
                incomplete_data.append(0.0)
        return Series(incomplete_data)

    def noise_all_series(self, power: float) -> dict:
        return self.deviate_all_series(
            {column: Deviation(self.add_noise, power) for column in SeriesColumn})

    def nullify_all_series(self, incomplete_part: float) -> dict:
        return self.deviate_all_series(
            {column: Deviation(self.add_incompleteness, incomplete_part) for column in SeriesColumn})

    def deviate_all_series(self, deviations: dict) -> dict:
        return {column: deviation.method(self.real_series[column], deviation.scale) for column, deviation in
                deviations.items()}

    def add_incompleteness_to_some_series_set(self, partially_incomplete_parts) -> dict:
        return {incomplete: self.add_incompleteness_to_some_series(
            {column: incompleted[incomplete] for column, incompleted in partially_incomplete_parts.items()})
            for incomplete in DeviationScale}

    def noise_some_series_set(self, partially_noised_strength) -> dict:
        return {strength: self.noise_some_series(
            {column: strengths[strength] for column, strengths in partially_noised_strength.items()})
            for strength in DeviationScale}

    def noise_some_series(self, noises: dict) -> dict:
        return self.deviate_some_series(
            {column: Deviation(self.add_noise, power) for column, power in noises.items()})

    def add_incompleteness_to_some_series(self, incomplete_parts: dict) -> dict:
        return self.deviate_some_series(
            {column: Deviation(self.add_incompleteness, incomplete_part) for column, incomplete_part in
             incomplete_parts.items()})

    def deviate_some_series(self, series_to_deviate: dict) -> dict:
        return {column: self.real_series[column] if column not in series_to_deviate.keys() else None
                for column in SeriesColumn} | \
            {column: deviation.method(self.real_series[column], deviation.scale)
             for column, deviation in series_to_deviate.items()}

    def get_deviated_series(self, source: DeviationSource,
                            deviation_range: DeviationRange = DeviationRange.ALL) -> dict:
        return self.all_deviated_series[source] if deviation_range == DeviationRange.ALL \
            else self.partially_deviated_series[source]

    def plot_single_series(self, data: Series, column: SeriesColumn, deviation: str = "", plot_type="-") -> None:
        plt.figure(figsize=(10, 4))
        plt.plot(data.values, plot_type)
        title = f"{self.company_name} {deviation} {column.value} prices"
        plt.title(title)
        plt.xlabel("Time [days]")
        plt.ylabel("Prices [USD]")
        save_image(plt, title)

    def plot_multiple_series(self, title: str, **kwargs) -> None:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111, axisbelow=True)
        for label, series in (kwargs.items()):
            ax.plot(series.values, markersize=1.5, label=label)
        title = f"{self.company_name} {title}"
        ax.set_title(title)
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Prices [USD]")
        set_legend(ax)
        save_image(plt, title)
        plt.show()
