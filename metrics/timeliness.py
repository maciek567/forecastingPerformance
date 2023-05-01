import math

from matplotlib import pyplot as plt

from timeseries.timeseries import StockMarketSeries
from timeseries.utils import DeviationScale, SeriesColumn, set_ticks_size, save_image


class HeinrichTimelinessMetric:

    def __init__(self, stock: StockMarketSeries):
        self.stock = stock

    @staticmethod
    def timeliness_values(decline: float, age: float) -> float:
        return math.exp(-decline * age)

    def timeliness_tuples(self, declines: dict, age: float) -> float:
        quality_sum = 0
        for column in SeriesColumn:
            quality_sum += self.timeliness_values(declines[column], age) * self.stock.weights[column]
        return quality_sum / sum(self.stock.weights.values())

    def timeliness_relations(self, declines: dict, ages_list: list) -> float:
        quality_sum = 0
        for i in range(len(ages_list)):
            quality_sum += self.timeliness_tuples(declines, ages_list[i])
        return quality_sum / len(ages_list)

    def values_qualities(self, decline: float, measurement_times: dict) -> tuple:
        deltas = {}
        qualities = {scale: [] for scale in DeviationScale}

        for scale in DeviationScale:
            deltas[scale], ages = self.stock.obsolescence.get_ages(measurement_times[scale])
            for i in range(self.stock.data_size):
                qualities[scale].append(self.timeliness_values(decline, ages[i]))

        return deltas, qualities

    def tuples_qualities(self, declines: dict, measurement_times: dict) -> tuple:
        deltas = {}
        qualities = {scale: [] for scale in DeviationScale}

        for scale in DeviationScale:
            deltas[scale], ages = self.stock.obsolescence.get_ages(measurement_times[scale])
            for i in range(self.stock.data_size):
                qualities[scale].append(self.timeliness_tuples(declines, ages[i]))

        return deltas, qualities

    def relation_qualities(self, declines: dict, measurement_times: dict) -> dict:
        ages = {scale: [] for scale in DeviationScale}

        for scale in DeviationScale:
            time_diffs, ages[scale] = self.stock.obsolescence.get_ages(measurement_times[scale])

        return {scale: self.timeliness_relations(declines, ages[scale]) for scale in DeviationScale}

    def draw_timeliness_qualities(self, times: dict, qualities: dict, column_name: SeriesColumn = None) -> None:
        column = column_name.value if column_name is not None else "all columns"
        fig, ax = plt.subplots(3, 1, figsize=(4, 5))
        fig.tight_layout(pad=0.5)
        plt.sca(ax[0])
        plt.xticks(*self.prepare_x_ticks(times[DeviationScale.SLIGHTLY]))
        plt.sca(ax[1])
        plt.xticks(*self.prepare_x_ticks(times[DeviationScale.MODERATELY]))
        plt.sca(ax[2])
        plt.xticks(*self.prepare_x_ticks(times[DeviationScale.HIGHLY]))
        ax[0].plot(times[DeviationScale.SLIGHTLY], qualities[DeviationScale.SLIGHTLY], color="b", markersize=2)
        ax[1].plot(times[DeviationScale.MODERATELY], qualities[DeviationScale.MODERATELY], color="g", markersize=2)
        ax[2].plot(times[DeviationScale.HIGHLY], qualities[DeviationScale.HIGHLY], color="r", markersize=2)
        title = f"Timeliness metric {column} prices"
        ax[0].set_title(title)
        ax[2].set_xlabel("Days before measurement")
        ax[1].set_ylabel("Quality")
        set_ticks_size(ax, "both", 10)
        save_image(plt, title)
        plt.show()

    @staticmethod
    def prepare_x_ticks(times):
        return [i for i in range(0, len(times), len(times) // 6)], [i for i in times if
                                                                    int(i) % (len(times) // 6) == 0]
