import math

from matplotlib import pyplot as plt

from timeseries.timeseries import StockMarketSeries
from timeseries.utils import DefectsScale, SeriesColumn


class HeinrichTimelinessMetric:

    def __init__(self, stock: StockMarketSeries):
        self.stock = stock

    @staticmethod
    def timeliness_values(decline: float, age: float) -> float:
        return math.exp(-decline * age)

    def timeliness_tuples(self, declines: dict, age: float, weights: dict) -> float:
        quality_sum = 0
        for column in SeriesColumn:
            quality_sum += self.timeliness_values(declines[column], age) * weights[column]
        return quality_sum / sum(weights.values())

    def timeliness_relations(self, declines: dict, ages_list: list, weights: dict) -> float:
        quality_sum = 0
        for i in range(len(ages_list)):
            quality_sum += self.timeliness_tuples(declines, ages_list[i], weights)
        return quality_sum / len(ages_list)

    def values_qualities(self, decline: float, measurement_times: dict) -> tuple:
        deltas = {}
        qualities = {scale: [] for scale in DefectsScale}

        for scale in DefectsScale:
            deltas[scale], ages = self.stock.get_ages(measurement_times[scale])
            for i in range(self.stock.time_series_end - self.stock.time_series_start):
                qualities[scale].append(self.timeliness_values(decline, ages[i]))

        return deltas, qualities

    def tuples_qualities(self, declines: dict, measurement_times: dict, weights: dict) -> tuple:
        deltas = {}
        qualities = {scale: [] for scale in DefectsScale}

        for scale in DefectsScale:
            deltas[scale], ages = self.stock.get_ages(measurement_times[scale])
            for i in range(self.stock.time_series_end - self.stock.time_series_start):
                qualities[scale].append(self.timeliness_tuples(declines, ages[i], weights))

        return deltas, qualities

    def relation_qualities(self, declines: dict, measurement_times: dict, weights: dict) -> dict:
        ages = {scale: [] for scale in DefectsScale}

        for scale in DefectsScale:
            time_diffs, ages[scale] = self.stock.get_ages(measurement_times[scale])

        return {scale: self.timeliness_relations(declines, ages[scale], weights) for scale in DefectsScale}

    def draw_timeliness_qualities(self, times: dict, qualities: dict, column_name: SeriesColumn = None) -> None:
        column = column_name.value if column_name is not None else "all columns"
        fig, ax = plt.subplots(3, 1)
        plt.sca(ax[0])
        plt.xticks(*self.prepare_x_ticks(times[DefectsScale.SLIGHTLY]))
        plt.sca(ax[1])
        plt.xticks(*self.prepare_x_ticks(times[DefectsScale.MODERATELY]))
        plt.sca(ax[2])
        plt.xticks(*self.prepare_x_ticks(times[DefectsScale.HIGHLY]))
        ax[0].plot(times[DefectsScale.SLIGHTLY], qualities[DefectsScale.SLIGHTLY], color="b", markersize=2)
        ax[1].plot(times[DefectsScale.MODERATELY], qualities[DefectsScale.MODERATELY], color="g", markersize=2)
        ax[2].plot(times[DefectsScale.HIGHLY], qualities[DefectsScale.HIGHLY], color="r", markersize=2)
        ax[0].set_title(f"Timeliness metric {column} prices")
        ax[2].set_xlabel("Days before measurement]")
        ax[1].set_ylabel("Quality")
        plt.show()

    @staticmethod
    def prepare_x_ticks(times):
        return [i for i in range(0, len(times), len(times) // 10)], [i for i in times if
                                                                     int(i) % (len(times) // 10) == 0]
