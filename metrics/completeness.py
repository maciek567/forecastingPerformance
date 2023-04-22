import math

import matplotlib.pyplot as plt

from metrics.utils import MetricLevel
from timeseries.timeseries import StockMarketSeries
from timeseries.utils import SeriesColumn, DeviationRange, DeviationScale, DeviationSource, save_image, set_ticks_size


class BlakeCompletenessMetric:

    def __init__(self, stock: StockMarketSeries):
        self.stock = stock

    @staticmethod
    def blake_values(value: float) -> float:
        return 1.0 if value != 0.0 else 0.0

    @staticmethod
    def blake_tuples(values: list) -> bool:
        return not (any(math.isclose(i, 0.0) for i in values))

    def blake_relation(self, relation: list) -> float:
        number_of_zero_tuples = 0
        for i in relation:
            if not self.blake_tuples(i):
                number_of_zero_tuples += 1
        return 1 - number_of_zero_tuples / len(relation)

    def values_qualities(self, column: SeriesColumn) -> dict:
        deviated_series = self.stock.get_deviated_series(DeviationSource.INCOMPLETENESS)
        qualities = {scale: [] for scale in DeviationScale}

        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            for scale in DeviationScale:
                qualities[scale].append(self.blake_values(deviated_series[scale][column][i]))

        return qualities

    def tuples_qualities(self, incompleteness_range: DeviationRange = DeviationRange.ALL) -> dict:
        deviated_series = self.stock.get_deviated_series(DeviationSource.INCOMPLETENESS, incompleteness_range)
        qualities = {scale: [] for scale in DeviationScale}

        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            for scale in DeviationScale:
                qualities[scale].append(self.blake_tuples(self.stock.get_list_for_tuple(deviated_series[scale], i)))

        return qualities

    def relation_qualities(self, incompleteness_range: DeviationRange = DeviationRange.ALL) -> dict:
        deviated_series = self.stock.get_deviated_series(DeviationSource.INCOMPLETENESS, incompleteness_range)
        deviated_tuples = {scale: [] for scale in DeviationScale}
        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            for scale in DeviationScale:
                deviated_tuples[scale].append(self.stock.get_list_for_tuple(deviated_series[scale], i))

        return {scale: self.blake_relation(deviated_tuples[scale]) for scale in DeviationScale}

    def draw_blake(self, qualities: dict, level: MetricLevel,
                   incompleteness_range: DeviationRange = DeviationRange.ALL,
                   column_name: SeriesColumn = None) -> None:
        fig, ax = plt.subplots(3, 1, figsize=(8, 4))

        range_title = ", " + incompleteness_range.value + " affected" if level == MetricLevel.TUPLES else ""
        column = column_name.value if column_name is not None else "all columns"
        title = f"Blake's metric {column} prices [{level.value}{range_title}]"
        fig.suptitle(title, size=12)
        ax[0].plot(qualities[DeviationScale.SLIGHTLY], "o", color="b", markersize=2)
        ax[1].plot(qualities[DeviationScale.MODERATELY], "o", color="g", markersize=2)
        ax[2].plot(qualities[DeviationScale.HIGHLY], "o", color="r", markersize=2)
        ax[0].set_title(f"Slightly incomplete (probability:"
                        f" {self.incompleteness_label(DeviationScale.SLIGHTLY, incompleteness_range)})", size=10)
        ax[1].set_title(f"Moderately incomplete probability:"
                        f" {self.incompleteness_label(DeviationScale.MODERATELY, incompleteness_range)}", size=10)
        ax[2].set_title(f"Highly incomplete probability:"
                        f" {self.incompleteness_label(DeviationScale.HIGHLY, incompleteness_range)}", size=10)
        ax[2].set_xlabel("Time [days]", size=10)
        ax[1].set_ylabel("Incompleteness metric value", size=10)
        set_ticks_size(ax, "x", 9)
        fig.tight_layout(pad=1.0)
        save_image(plt, title)
        plt.show()

    def incompleteness_label(self, incompleteness: DeviationScale, incompleteness_range: DeviationRange):
        if incompleteness_range == DeviationRange.ALL:
            return self.stock.incompleteness.all_incomplete_parts[incompleteness]
        else:
            return str({column.value: probabilities[incompleteness] for column, probabilities in
                        self.stock.incompleteness.partially_incomplete_parts.items()}).replace("\'", "")
