import math

import matplotlib.pyplot as plt

from metrics.utils import MetricLevel, DefectionRange, DefectsScale, DefectsSource
from timeseries.timeseries import StockMarketSeries
from timeseries.utils import SeriesColumn


class BlakeCompletenessMetric:

    def __init__(self, stock: StockMarketSeries):
        self.stock = stock

    @staticmethod
    def blake_values(value: float):
        return 1.0 if value != 0.0 else 0.0

    @staticmethod
    def blake_tuples(values: list):
        return not (any(math.isclose(i, 0.0) for i in values))

    def blake_relation(self, relation: list):
        number_of_zero_tuples = 0
        for i in relation:
            if not self.blake_tuples(i):
                number_of_zero_tuples += 1
        return 1 - number_of_zero_tuples / len(relation)

    def values_qualities(self, column: SeriesColumn):
        series = self.init_series()
        qualities_first, qualities_second, qualities_third = [], [], []
        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            qualities_first.append(self.blake_values(series[DefectsScale.SLIGHTLY][column][i]))
            qualities_second.append(self.blake_values(series[DefectsScale.MODERATELY][column][i]))
            qualities_third.append(self.blake_values(series[DefectsScale.HIGHLY][column][i]))
        return qualities_first, qualities_second, qualities_third

    def tuples_qualities(self, incompleteness_range: DefectionRange = DefectionRange.ALL):
        series = self.init_series(incompleteness_range)
        qualities_first, qualities_second, qualities_third = [], [], []

        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            tuple_slightly_incomplete = self.stock.create_tuple(series[DefectsScale.SLIGHTLY], i)
            tuple_moderately_incomplete = self.stock.create_tuple(series[DefectsScale.MODERATELY], i)
            tuple_highly_incomplete = self.stock.create_tuple(series[DefectsScale.HIGHLY], i)
            qualities_first.append(self.blake_tuples(tuple_slightly_incomplete))
            qualities_second.append(self.blake_tuples(tuple_moderately_incomplete))
            qualities_third.append(self.blake_tuples(tuple_highly_incomplete))

        return qualities_first, qualities_second, qualities_third

    def relation_qualities(self, incompleteness_range: DefectionRange = DefectionRange.ALL):
        series = self.init_series(incompleteness_range)
        tuples_slightly_incomplete, tuples_moderately_incomplete, tuples_highly_incomplete = [], [], []
        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            tuples_slightly_incomplete.append(self.stock.create_tuple(series[DefectsScale.SLIGHTLY], i))
            tuples_moderately_incomplete.append(self.stock.create_tuple(series[DefectsScale.MODERATELY], i))
            tuples_highly_incomplete.append(self.stock.create_tuple(series[DefectsScale.HIGHLY], i))
        return self.blake_relation(tuples_slightly_incomplete), self.blake_relation(
            tuples_moderately_incomplete), self.blake_relation(tuples_highly_incomplete)

    def init_series(self, incompleteness_range: DefectionRange = None):
        return self.stock.partially_defected_series[DefectsSource.INCOMPLETENESS] \
            if incompleteness_range == DefectionRange.PARTIAL \
            else self.stock.all_defected_series[DefectsSource.INCOMPLETENESS]

    def draw_blake(self, slightly: list, moderately: list, highly: list, level: MetricLevel,
                   incompleteness_range: DefectionRange = DefectionRange.ALL,
                   column_name: SeriesColumn = None):
        fig, ax = plt.subplots(3, 1)
        range_title = ", " + incompleteness_range.value + " affected" if level == MetricLevel.TUPLES else ""
        column = column_name.value if column_name is not None else "all columns"
        fig.suptitle(f"Blake's metric {column} prices [{level.value}{range_title}]", size=14)
        ax[0].plot(slightly, "o", color="b", markersize=2)
        ax[1].plot(moderately, "o", color="g", markersize=2)
        ax[2].plot(highly, "o", color="r", markersize=2)
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
