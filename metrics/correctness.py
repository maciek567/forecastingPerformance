import matplotlib.pyplot as plt
from pandas import Series

from metrics.utils import Sensitiveness, Strength, QualityDifferencesSource, MetricLevel
from timeseries.timeseries import StockMarketSeries
from timeseries.utils import SeriesColumn


class HeinrichCorrectnessMetric:

    def __init__(self, alpha: dict = None):
        if alpha is None:
            self.alpha = {Sensitiveness.SENSITIVE: 1.0, Sensitiveness.MODERATE: 1.0, Sensitiveness.INSENSITIVE: 1.0}
        else:
            self.alpha = alpha

    @staticmethod
    def d(w_i: float, w_r: float, alpha: float):
        return pow(abs(w_i - w_r) / max(abs(w_i), abs(w_r)), alpha)

    def heinrich_values(self, w_i: float, w_r: float, alpha: float):
        return 1 - self.d(w_i, w_r, alpha)

    def heinrich_tuples(self, t: list, e: list, g: list, alpha: float = 1.0):
        quality_sum = 0
        for i in range(len(t)):
            quality_sum += self.heinrich_values(t[i], e[i], alpha) * g[i]
        return quality_sum / sum(g)

    def heinrich_relation(self, r: list, e: list, tuple_weights: list, alpha: float = 1.0):
        quality_sum = 0
        for i in range(len(r)):
            quality_sum += self.heinrich_tuples(r[i], e[i], tuple_weights, alpha)
        return quality_sum / len(r)

    def values_qualities(self, stock: StockMarketSeries, column: SeriesColumn,
                         is_alpha: bool = True, are_different_noises: bool = False):
        alpha, first_series, second_series, third_series = self.init_alpha_and_series(are_different_noises, is_alpha,
                                                                                      stock)
        first_quality, second_quality, third_quality = [], [], []

        for i in range(stock.time_series_start, stock.time_series_end):
            value = stock.series[column][i]
            first_quality.append(self.heinrich_values(first_series[column][i], value, alpha[Sensitiveness.SENSITIVE]))
            second_quality.append(self.heinrich_values(second_series[column][i], value, alpha[Sensitiveness.MODERATE]))
            third_quality.append(self.heinrich_values(third_series[column][i], value, alpha[Sensitiveness.INSENSITIVE]))

        return first_quality, second_quality, third_quality

    def tuples_qualities(self, stock: StockMarketSeries, tuple_weights: list,
                         is_alpha: bool = True, are_different_noises: bool = False):
        alpha, first_series, second_series, third_series = self.init_alpha_and_series(are_different_noises, is_alpha,
                                                                                      stock)
        first_quality, second_quality, third_quality = [], [], []

        for i in range(stock.time_series_start, stock.time_series_end):
            tuple = self.create_tuple(stock.series, i)
            first_tuple = self.create_tuple(first_series, i)
            second_tuple = self.create_tuple(second_series, i)
            third_tuple = self.create_tuple(third_series, i)

            first_quality.append(
                self.heinrich_tuples(first_tuple, tuple, tuple_weights, alpha[Sensitiveness.SENSITIVE]))
            second_quality.append(
                self.heinrich_tuples(second_tuple, tuple, tuple_weights, alpha[Sensitiveness.MODERATE]))
            third_quality.append(
                self.heinrich_tuples(third_tuple, tuple, tuple_weights, alpha[Sensitiveness.INSENSITIVE]))

        return first_quality, second_quality, third_quality

    def relation_qualities(self, stock: StockMarketSeries, tuple_weights: list,
                           is_alpha: bool = True, are_different_noises: bool = False):
        alpha, first_series, second_series, third_series = self.init_alpha_and_series(are_different_noises, is_alpha,
                                                                                      stock)
        tuples, tuples_noised_first, tuples_noised_second, tuples_noised_third = [], [], [], []

        for i in range(stock.time_series_start, stock.time_series_end):
            tuples.append(self.create_tuple(stock.series, i))
            tuples_noised_first.append(self.create_tuple(first_series, i))
            tuples_noised_second.append(self.create_tuple(second_series, i))
            tuples_noised_third.append(self.create_tuple(third_series, i))
        return self.heinrich_relation(tuples_noised_first, tuples, tuple_weights, alpha[Sensitiveness.SENSITIVE]), \
            self.heinrich_relation(tuples_noised_second, tuples, tuple_weights, alpha[Sensitiveness.MODERATE]), \
            self.heinrich_relation(tuples_noised_third, tuples, tuple_weights, alpha[Sensitiveness.INSENSITIVE])

    @staticmethod
    def create_tuple(series: Series, i: int):
        return [series[SeriesColumn.OPEN][i], series[SeriesColumn.CLOSE][i],
                series[SeriesColumn.ADJ_CLOSE][i], series[SeriesColumn.HIGH][i],
                series[SeriesColumn.LOW][i], series[SeriesColumn.VOLUME][i]]

    def init_alpha_and_series(self, are_different_noises: bool, is_alpha: bool, stock: StockMarketSeries):
        alpha = {Sensitiveness.SENSITIVE: 1.0, Sensitiveness.MODERATE: 1.0, Sensitiveness.INSENSITIVE: 1.0}
        if is_alpha:
            alpha = self.alpha
        first_series, second_series, third_series = stock.series_noised[Strength.MODERATE], \
            stock.series_noised[Strength.MODERATE], stock.series_noised[Strength.MODERATE]
        if are_different_noises:
            first_series, second_series, third_series = stock.series_noised[Strength.WEAK], \
                stock.series_noised[Strength.MODERATE], stock.series_noised[Strength.STRONG]
        return alpha, first_series, second_series, third_series

    def draw_heinrich_qualities(self, stock: StockMarketSeries, first: list, second: list, third: list,
                                quality_differences: QualityDifferencesSource, level: MetricLevel = MetricLevel.VALUES):
        fig = plt.figure(facecolor="w")
        ax = fig.add_subplot(111, facecolor="#dddddd", axisbelow=True)
        if quality_differences == QualityDifferencesSource.NOISE_STRENGTH:
            ax.plot(first, "b", lw=1, label=f"Weak noise: std={stock.noises_strength[Strength.WEAK]}")
            ax.plot(second, "r", lw=1, label=f"Medium noise: std={stock.noises_strength[Strength.MODERATE]}")
            ax.plot(third, "g", lw=1, label=f"Strong noise: std={stock.noises_strength[Strength.STRONG]}")
        elif quality_differences == QualityDifferencesSource.SENSITIVENESS:
            ax.plot(first, "b", lw=1, label=f"Sensitive: alpha={self.alpha[Sensitiveness.SENSITIVE]}")
            ax.plot(second, "r", lw=1, label=f"Medium sensitivity: alpha={self.alpha[Sensitiveness.MODERATE]}")
            ax.plot(third, "g", lw=1, label=f"Insensitive: alpha={self.alpha[Sensitiveness.INSENSITIVE]}")
        ax.set_title(f"Heinrich quality metric [different {quality_differences.value}, {level.value} level]")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Quality")
        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.2))
        legend.get_frame().set_alpha(0.5)
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(False)
        plt.show()
