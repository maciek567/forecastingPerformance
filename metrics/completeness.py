import math
import matplotlib.pyplot as plt

from timeseries.timeseries import StockMarketSeries
from timeseries.utils import SeriesColumn
from metrics.utils import Incomplete, MetricLevel


class BlakeCompletenessMetric:

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

    def values_qualities(self, stock: StockMarketSeries, column: SeriesColumn):
        qualities_first, qualities_second, qualities_third = [], [], []
        for i in range(stock.time_series_end - stock.time_series_start):
            qualities_first.append(self.blake_values(stock.series_incomplete[Incomplete.SLIGHTLY][column][i]))
            qualities_second.append(self.blake_values(stock.series_incomplete[Incomplete.MODERATELY][column][i]))
            qualities_third.append(self.blake_values(stock.series_incomplete[Incomplete.HIGHLY][column][i]))
        return qualities_first, qualities_second, qualities_third

    def tuples_qualities(self, stock: StockMarketSeries):
        qualities_first, qualities_second, qualities_third = [], [], []

        for i in range(stock.time_series_end - stock.time_series_start):
            tuple_slightly_incomplete = self.create_tuple(stock.series_incomplete[Incomplete.SLIGHTLY], i)
            tuple_moderately_incomplete = self.create_tuple(stock.series_incomplete[Incomplete.MODERATELY], i)
            tuple_highly_incomplete = self.create_tuple(stock.series_incomplete[Incomplete.HIGHLY], i)
            qualities_first.append(self.blake_tuples(tuple_slightly_incomplete))
            qualities_second.append(self.blake_tuples(tuple_moderately_incomplete))
            qualities_third.append(self.blake_tuples(tuple_highly_incomplete))

        return qualities_first, qualities_second, qualities_third

    def relation_qualities(self, stock: StockMarketSeries):
        tuples_slightly_incomplete, tuples_moderately_incomplete, tuples_highly_incomplete = [], [], []
        for i in range(stock.time_series_end - stock.time_series_start):
            tuples_slightly_incomplete.append(self.create_tuple(stock.series_incomplete[Incomplete.SLIGHTLY], i))
            tuples_moderately_incomplete.append(self.create_tuple(stock.series_incomplete[Incomplete.MODERATELY], i))
            tuples_highly_incomplete.append(self.create_tuple(stock.series_incomplete[Incomplete.HIGHLY], i))
        return self.blake_relation(tuples_slightly_incomplete), self.blake_relation(
            tuples_moderately_incomplete), self.blake_relation(tuples_highly_incomplete)

    @staticmethod
    def create_tuple(series: dict, i: int):
        return [series[SeriesColumn.OPEN][i],
                series[SeriesColumn.CLOSE][i],
                series[SeriesColumn.ADJ_CLOSE][i],
                series[SeriesColumn.HIGH][i],
                series[SeriesColumn.LOW][i],
                series[SeriesColumn.VOLUME][i]]

    @staticmethod
    def draw_blake(stock: StockMarketSeries, slightly: list, moderately: list, highly: list, level: MetricLevel):
        fig, ax = plt.subplots(3, 1)
        fig.suptitle(f"Blake quality metric [{level.value} level]: {stock.company_name}", size=14)
        ax[0].plot(slightly, "o", color="b", markersize=2)
        ax[1].plot(moderately, "o", color="g", markersize=2)
        ax[2].plot(highly, "o", color="r", markersize=2)
        ax[0].set_title(f"Slightly incomplete (probability: {stock.incomplete_parts[Incomplete.SLIGHTLY]})", size=10)
        ax[1].set_title(f"Moderately incomplete probability: {stock.incomplete_parts[Incomplete.MODERATELY]}", size=10)
        ax[2].set_title(f"Highly incomplete probability: {stock.incomplete_parts[Incomplete.HIGHLY]}", size=10)
        ax[2].set_xlabel("Time [days]", size=10)
        ax[1].set_ylabel("Incompleteness metric value", size=10)
        fig.tight_layout(pad=1.0)
        plt.show()
