import matplotlib.pyplot as plt

from inout.paths import metrics_graphs_path
from metrics.utils import MetricLevel
from timeseries.enums import SeriesColumn, DeviationRange, DeviationSource, DeviationScale, Mitigation
from timeseries.timeseries import StockMarketSeries
from util.graphs import save_image, set_legend, TIME_DAYS_LABEL


class HeinrichCorrectnessMetric:

    def __init__(self, stock: StockMarketSeries, alpha: dict = None):
        self.stock = stock
        self.default_alpha = {column: 0.5 for column in SeriesColumn}
        self.custom_alpha = alpha

    @staticmethod
    def d(noised: float, real: float, alpha: float) -> float:
        return pow(abs(noised - real) / max(abs(noised), abs(real)), alpha)

    def heinrich_values(self, noised: float, real: float, alpha: float) -> float:
        return 1 - self.d(noised, real, alpha)

    def heinrich_tuples(self, noised: dict, real: dict, alpha: dict) -> float:
        quality_sum = 0
        weights = self.stock.weights
        for column in SeriesColumn:
            quality_sum += self.heinrich_values(noised[column], real[column], alpha[column]) * weights[column]
        return quality_sum / sum(weights.values())

    def heinrich_relation(self, noised: list, real: list, alpha: dict) -> float:
        quality_sum = 0
        for i in range(len(noised)):
            quality_sum += self.heinrich_tuples(noised[i], real[i], alpha)
        return quality_sum / len(noised)

    def values_qualities(self, column: SeriesColumn, is_alpha: bool = True) -> dict:
        alpha = self.get_alpha(is_alpha)
        deviated_series = self.stock.get_deviated_series(DeviationSource.NOISE)
        qualities = {scale: [] for scale in DeviationScale}

        for i in range(self.stock.data_size):
            value = self.stock.real_series[column][i]
            for scale in DeviationScale:
                qualities[scale].append(self.heinrich_values(deviated_series[scale][column][i], value, alpha[column]))

        return qualities

    def tuples_qualities(self, noise_range: DeviationRange = DeviationRange.ALL, is_alpha: bool = True):
        alpha = self.get_alpha(is_alpha)
        deviated_series = self.stock.get_deviated_series(DeviationSource.NOISE, noise_range)
        qualities = {scale: [] for scale in DeviationScale}

        for i in range(self.stock.data_size):
            real_tuple = self.stock.get_dict_for_tuple(self.stock.real_series, i)
            for scale in DeviationScale:
                qualities[scale].append(self.heinrich_tuples(self.stock.get_dict_for_tuple(deviated_series[scale], i),
                                                             real_tuple, alpha))

        return qualities

    def relation_qualities(self, noise_range: DeviationRange, is_alpha: bool = True) -> dict:
        alpha = self.get_alpha(is_alpha)
        deviated_series = self.stock.get_deviated_series(DeviationSource.NOISE, noise_range)
        mitigated_series = self.stock.get_mitigated_series(DeviationSource.NOISE, noise_range)
        real_tuples = []
        deviated_tuples = {scale: [] for scale in DeviationScale}
        mitigated_tuples = {scale: [] for scale in DeviationScale}

        for i in range(self.stock.data_size):
            real_tuples.append(self.stock.get_dict_for_tuple(self.stock.real_series, i))
            for scale in DeviationScale:
                deviated_tuples[scale].append(self.stock.get_dict_for_tuple(deviated_series[scale], i))
                mitigated_tuples[scale].append(self.stock.get_dict_for_tuple(mitigated_series[scale], i))

        return {scale: {Mitigation.NOT_MITIGATED: self.heinrich_relation(deviated_tuples[scale], real_tuples, alpha),
                        Mitigation.MITIGATED: self.heinrich_relation(mitigated_tuples[scale], real_tuples, alpha)}
                for scale in DeviationScale}

    def get_alpha(self, is_alpha: bool) -> dict:
        return self.custom_alpha if is_alpha else self.default_alpha

    def draw_heinrich_qualities(self, qualities: dict,
                                level: MetricLevel, is_alpha: bool,
                                noise_range: DeviationRange = DeviationRange.ALL,
                                column_name: SeriesColumn = None) -> None:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111, axisbelow=True)
        ax.plot(qualities[DeviationScale.SLIGHTLY], "b", lw=1,
                label=f"Weak noise: std={self.noises_label(DeviationScale.SLIGHTLY, noise_range)}")
        ax.plot(qualities[DeviationScale.MODERATELY], "r", lw=1,
                label=f"Medium noise: std={self.noises_label(DeviationScale.MODERATELY, noise_range)}")
        ax.plot(qualities[DeviationScale.HIGHLY], "g", lw=1,
                label=f"Strong noise: std={self.noises_label(DeviationScale.HIGHLY, noise_range)}")
        column = column_name.value if column_name is not None else "all columns"
        noise = noise_range.value if noise_range is not None else "all"
        sensitiveness = ", sensitiveness" if is_alpha else ""
        title = f"Heinrich metric {column} prices [{level.value}, {noise} noised{sensitiveness}]"
        ax.set_title(title)
        ax.set_xlabel(TIME_DAYS_LABEL)
        ax.set_ylabel("Quality")
        set_legend(ax)
        save_image(plt, title, metrics_graphs_path)
        plt.show()

    def noises_label(self, strength: DeviationScale, noise_range: DeviationRange) -> str:
        if noise_range == DeviationRange.ALL:
            return str(self.stock.noises.all_noises_strength[strength])
        else:
            return "\n" + str({column.value: strengths[strength] for column, strengths in
                               self.stock.noises.partially_noised_strength.items()}).replace("\'", "")

    @staticmethod
    def get_deviation_name():
        return DeviationSource.NOISE
