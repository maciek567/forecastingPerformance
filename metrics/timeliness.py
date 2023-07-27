import math

from matplotlib import pyplot as plt

from inout.paths import metrics_graphs_path
from timeseries.enums import DeviationScale, SeriesColumn, Mitigation, DeviationRange, DeviationSource
from timeseries.timeseries import StockMarketSeries
from timeseries.utils import normalize_with_columns
from util.graphs import TIME_DAYS_LABEL, METRIC_SCORE, save_image, set_legend


class HeinrichTimelinessMetric:

    def __init__(self, stock: StockMarketSeries, declines: dict, measurement_times: dict):
        self.stock = stock
        self.declines = declines
        self.measurement_times = measurement_times

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

    def values_qualities(self, column: SeriesColumn) -> tuple:
        deltas = {}
        qualities = {scale: [] for scale in DeviationScale}

        for scale in DeviationScale:
            deltas[scale], ages = self.stock.obsolescence.get_ages(self.measurement_times[scale])
            for i in range(self.stock.data_size):
                qualities[scale].append(self.timeliness_values(self.declines[column], ages[i]))

        return deltas, qualities

    def tuples_qualities(self) -> tuple:
        deltas = {}
        qualities = {scale: [] for scale in DeviationScale}

        for scale in DeviationScale:
            deltas[scale], ages = self.stock.obsolescence.get_ages(self.measurement_times[scale])
            for i in range(self.stock.data_size):
                qualities[scale].append(self.timeliness_tuples(self.declines, ages[i]))

        return deltas, qualities

    def relation_qualities(self, obsoleteness_range: DeviationRange, columns: list = None) -> dict:
        if obsoleteness_range == DeviationRange.ALL:
            if columns != [] and columns is not None:
                raise Exception("Columns can be specified only for partial range.")

            ages = {scale: [] for scale in DeviationScale}

            for scale in DeviationScale:
                time_diffs, ages[scale] = self.stock.obsolescence.get_ages(self.measurement_times[scale])

            return {scale: {Mitigation.NOT_MITIGATED: self.timeliness_relations(self.declines, ages[scale])}
                    for scale in DeviationScale}

        else:
            columns = self.stock.columns if (columns is None or columns == []) else columns
            ages = {scale: {column: [] for column in SeriesColumn} for scale in DeviationScale}
            real, deviated = self.stock.determine_real_and_deviated_columns(DeviationRange.PARTIAL,
                                                                            DeviationSource.TIMELINESS, columns)
            normalized_weights = normalize_with_columns(self.stock.weights, columns)

            for scale in DeviationScale:
                for column in real:
                    time_diffs, ages[scale][column] = self.stock.obsolescence.get_ages(0)
                for column in deviated:
                    time_diffs, ages[scale][column] = self.stock.obsolescence.get_ages(self.measurement_times[scale])

            return {scale: {Mitigation.NOT_MITIGATED: sum(
                [self.timeliness_relations(self.declines, ages[scale][column]) * normalized_weights[column]
                 for column in columns])}
                for scale in DeviationScale}

    @staticmethod
    def draw_timeliness_qualities(times: dict, qualities: dict, column_name: SeriesColumn = None) -> None:
        column = column_name.value if column_name is not None else "all columns"
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111, axisbelow=True)
        graph_size = len(times[DeviationScale.SLIGHTLY])
        slightly_start = int(times[DeviationScale.HIGHLY][0]) - int(times[DeviationScale.SLIGHTLY][0])
        moderately_start = int(times[DeviationScale.HIGHLY][0]) - int(times[DeviationScale.MODERATELY][0])

        ax.plot([i for i in range(slightly_start, slightly_start + graph_size)],
                qualities[DeviationScale.SLIGHTLY], color="b", markersize=2,
                label=f"Slightly outdated: {int(times[DeviationScale.SLIGHTLY][-1]) - 1} days")
        ax.plot([i for i in range(moderately_start, moderately_start + graph_size)],
                qualities[DeviationScale.MODERATELY], color="g", markersize=2,
                label=f"Moderately outdated: {int(times[DeviationScale.MODERATELY][-1]) - 1} days")
        ax.plot([i for i in range(0, graph_size)],
                qualities[DeviationScale.HIGHLY], color="r", markersize=2,
                label=f"Highly outdated: {int(times[DeviationScale.HIGHLY][-1]) - 1} days")

        ax.axvline(x=int(times[DeviationScale.HIGHLY][0]), color='black', label='Metric calculation timestamp',
                   linestyle="--", linewidth='1')

        title = f"Timeliness metric {column} prices"
        ax.set_title(title)
        ax.set_xlabel(TIME_DAYS_LABEL)
        ax.set_ylabel(METRIC_SCORE)
        set_legend(ax)
        save_image(plt, title, metrics_graphs_path)
        plt.show()

    @staticmethod
    def get_deviation_name():
        return DeviationSource.TIMELINESS
