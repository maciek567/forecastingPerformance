import math

import matplotlib.pyplot as plt

from metrics.utils import MetricLevel
from timeseries.timeseries import StockMarketSeries
from timeseries.utils import SeriesColumn, DefectionRange, DefectsScale, DefectsSource


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
        defected_series = self.stock.get_defected_series(DefectsSource.INCOMPLETENESS)
        qualities = {scale: [] for scale in DefectsScale}

        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            for scale in DefectsScale:
                qualities[scale].append(self.blake_values(defected_series[scale][column][i]))

        return qualities

    def tuples_qualities(self, incompleteness_range: DefectionRange = DefectionRange.ALL) -> dict:
        defected_series = self.stock.get_defected_series(DefectsSource.INCOMPLETENESS, incompleteness_range)
        qualities = {scale: [] for scale in DefectsScale}

        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            for scale in DefectsScale:
                qualities[scale].append(self.blake_tuples(self.stock.create_tuple(defected_series[scale], i)))

        return qualities

    def relation_qualities(self, incompleteness_range: DefectionRange = DefectionRange.ALL) -> dict:
        defected_series = self.stock.get_defected_series(DefectsSource.INCOMPLETENESS, incompleteness_range)
        defected_tuples = {scale: [] for scale in DefectsScale}
        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            for scale in DefectsScale:
                defected_tuples[scale].append(self.stock.create_tuple(defected_series[scale], i))

        return {scale: self.blake_relation(defected_tuples[scale]) for scale in DefectsScale}

    def draw_blake(self, qualities: dict, level: MetricLevel,
                   incompleteness_range: DefectionRange = DefectionRange.ALL,
                   column_name: SeriesColumn = None) -> None:
        fig, ax = plt.subplots(3, 1)
        range_title = ", " + incompleteness_range.value + " affected" if level == MetricLevel.TUPLES else ""
        column = column_name.value if column_name is not None else "all columns"
        fig.suptitle(f"Blake's metric {column} prices [{level.value}{range_title}]", size=14)
        ax[0].plot(qualities[DefectsScale.SLIGHTLY], "o", color="b", markersize=2)
        ax[1].plot(qualities[DefectsScale.MODERATELY], "o", color="g", markersize=2)
        ax[2].plot(qualities[DefectsScale.HIGHLY], "o", color="r", markersize=2)
        ax[0].set_title(f"Slightly incomplete (probability:"
                        f" {self.incompleteness_label(DefectsScale.SLIGHTLY, incompleteness_range)})", size=10)
        ax[1].set_title(f"Moderately incomplete probability:"
                        f" {self.incompleteness_label(DefectsScale.MODERATELY, incompleteness_range)}", size=10)
        ax[2].set_title(f"Highly incomplete probability:"
                        f" {self.incompleteness_label(DefectsScale.HIGHLY, incompleteness_range)}", size=10)
        ax[2].set_xlabel("Time [days]", size=10)
        ax[1].set_ylabel("Incompleteness metric value", size=10)
        fig.tight_layout(pad=1.0)
        plt.show()

    def incompleteness_label(self, incompleteness: DefectsScale, incompleteness_range: DefectionRange):
        if incompleteness_range == DefectionRange.ALL:
            return self.stock.all_incomplete_parts[incompleteness]
        else:
            return str({column.value: probabilities[incompleteness] for column, probabilities in
                        self.stock.partially_incomplete_parts.items()})
