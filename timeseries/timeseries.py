import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series

from metrics.utils import DefectsScale, DefectsSource
from timeseries.utils import SeriesColumn


class Defection:
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
            else {DefectsScale.SLIGHTLY: 0.4, DefectsScale.MODERATELY: 1.0, DefectsScale.HIGHLY: 3.0}
        self.all_incomplete_parts = all_incomplete_parts if all_incomplete_parts is not None \
            else {DefectsScale.SLIGHTLY: 0.05, DefectsScale.MODERATELY: 0.12, DefectsScale.HIGHLY: 0.3}
        self.all_defected_series = {
            DefectsSource.NOISE:
                {strength: self.noise_all_series(self.all_noises_strength[strength]) for strength in DefectsScale},
            DefectsSource.INCOMPLETENESS:
                {strength: self.nullify_all_series(self.all_incomplete_parts[strength]) for strength in DefectsScale}}
        self.partially_defected_series = {}
        if partially_noised_strength is not None:
            self.partially_noised_strength = partially_noised_strength
            self.partially_defected_series[DefectsSource.NOISE] = self.noise_some_series_set(partially_noised_strength)
        if partially_incomplete_parts is not None:
            self.partially_incomplete_parts = partially_incomplete_parts
            self.partially_defected_series[DefectsSource.INCOMPLETENESS] = \
                self.add_incompleteness_to_some_series_set(partially_incomplete_parts)

    def create_single_series(self, column_name: SeriesColumn):
        series = pd.Series(list(self.data[column_name]), index=self.data["date"])
        return series[self.time_series_start:self.time_series_end]

    def create_multiple_series(self):
        return {column: self.create_single_series(column.value) for column in SeriesColumn}

    @staticmethod
    def create_tuple(series: dict, i: int):
        return [series[column][i] for column in SeriesColumn]

    def add_noise(self, data: Series, power: float):
        mean = 0
        std_dev = power
        noise = np.random.normal(mean, std_dev, self.time_series_end - self.time_series_start)
        return data + noise

    def add_incompleteness(self, data: Series, incomplete_part: float):
        incompleteness = np.random.choice([0, 1], self.time_series_end - self.time_series_start,
                                          p=[incomplete_part, 1.0 - incomplete_part])
        incomplete_data = []
        for i in range(0, self.time_series_end - self.time_series_start):
            if incompleteness[i] == 1:
                incomplete_data.append(data[i])
            else:
                incomplete_data.append(0.0)
        return Series(incomplete_data)

    def noise_all_series(self, power: float):
        return self.defect_all_series(
            {column: Defection(self.add_noise, power) for column in SeriesColumn})

    def nullify_all_series(self, incomplete_part: float):
        return self.defect_all_series(
            {column: Defection(self.add_incompleteness, incomplete_part) for column in SeriesColumn})

    def defect_all_series(self, defections: dict):
        return {column: defection.method(self.real_series[column], defection.scale) for column, defection in
                defections.items()}

    def add_incompleteness_to_some_series_set(self, partially_incomplete_parts):
        return {incomplete: self.add_incompleteness_to_some_series(
            {column: incompleted[incomplete] for column, incompleted in partially_incomplete_parts.items()})
            for incomplete in DefectsScale}

    def noise_some_series_set(self, partially_noised_strength):
        return {strength: self.noise_some_series(
            {column: strengths[strength] for column, strengths in partially_noised_strength.items()})
            for strength in DefectsScale}

    def noise_some_series(self, noises: dict):
        return self.defect_some_series(
            {column: Defection(self.add_noise, power) for column, power in noises.items()})

    def add_incompleteness_to_some_series(self, incomplete_parts: dict):
        return self.defect_some_series(
            {column: Defection(self.add_incompleteness, incomplete_part) for column, incomplete_part in
             incomplete_parts.items()})

    def defect_some_series(self, series_to_defect: dict):
        return {column: self.real_series[column] if column not in series_to_defect.keys() else None
                for column in SeriesColumn} | \
            {column: defection.method(self.real_series[column], defection.scale)
             for column, defection in series_to_defect.items()}

    def plot_single_series(self, data: Series, column: SeriesColumn, defection: str = "", plot_type="-"):
        plt.figure(figsize=(10, 4))
        plt.plot(data.values, plot_type)
        plt.title(f"{self.company_name} {defection} {column.value} price")
        plt.xlabel("Time [days]")
        plt.ylabel("Prices [USD]")

    def plot_multiple_series(self, title: str, **kwargs):
        fig = plt.figure(facecolor="w", figsize=(10, 4))
        ax = fig.add_subplot(111, facecolor="#dddddd", axisbelow=True)
        for label, series in (kwargs.items()):
            ax.plot(series.values, markersize=1.5, label=label)
        ax.set_title(f"{self.company_name} {title}")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Prices [USD]")
        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.2))
        legend.get_frame().set_alpha(0.5)
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(False)
        plt.show()
