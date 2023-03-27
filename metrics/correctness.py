import matplotlib.pyplot as plt

from metrics.utils import Strength, MetricLevel, DefectionRange
from timeseries.timeseries import StockMarketSeries
from timeseries.utils import SeriesColumn


class HeinrichCorrectnessMetric:

    def __init__(self, stock: StockMarketSeries, alpha: dict = None):
        self.stock = stock
        self.default_alpha = {column: 1.0 for column in SeriesColumn}
        self.custom_alpha = alpha

    @staticmethod
    def d(w_i: float, w_r: float, alpha: float):
        return pow(abs(w_i - w_r) / max(abs(w_i), abs(w_r)), alpha)

    def heinrich_values(self, w_i: float, w_r: float, alpha: float):
        return 1 - self.d(w_i, w_r, alpha)

    def heinrich_tuples(self, t: list, e: list, g: list, alpha: list):
        quality_sum = 0
        for i in range(len(t)):
            quality_sum += self.heinrich_values(t[i], e[i], alpha[i]) * g[i]
        return quality_sum / sum(g)

    def heinrich_relation(self, r: list, e: list, tuple_weights: list, alpha: list):
        quality_sum = 0
        for i in range(len(r)):
            quality_sum += self.heinrich_tuples(r[i], e[i], tuple_weights, alpha)
        return quality_sum / len(r)

    def values_qualities(self, column: SeriesColumn,
                         is_alpha: bool = True):
        alpha = self.init_alpha(is_alpha)
        first_series, second_series, third_series = self.init_series()
        first_quality, second_quality, third_quality = [], [], []

        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            value = self.stock.real_series[column][i]
            first_quality.append(self.heinrich_values(first_series[column][i], value, alpha[column]))
            second_quality.append(self.heinrich_values(second_series[column][i], value, alpha[column]))
            third_quality.append(self.heinrich_values(third_series[column][i], value, alpha[column]))

        return first_quality, second_quality, third_quality

    def tuples_qualities(self, tuple_weights: list, noised_series: DefectionRange = DefectionRange.ALL,
                         is_alpha: bool = True):
        alpha = self.init_alpha(is_alpha)
        first_series, second_series, third_series = self.init_series(noised_series=noised_series)
        first_quality, second_quality, third_quality = [], [], []

        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            tuple = self.stock.create_tuple(self.stock.real_series, i)
            first_tuple = self.stock.create_tuple(first_series, i)
            second_tuple = self.stock.create_tuple(second_series, i)
            third_tuple = self.stock.create_tuple(third_series, i)

            first_quality.append(
                self.heinrich_tuples(first_tuple, tuple, tuple_weights, list(alpha.values())))
            second_quality.append(
                self.heinrich_tuples(second_tuple, tuple, tuple_weights, list(alpha.values())))
            third_quality.append(
                self.heinrich_tuples(third_tuple, tuple, tuple_weights, list(alpha.values())))

        return first_quality, second_quality, third_quality

    def relation_qualities(self, tuple_weights: list, noised_series: DefectionRange = DefectionRange.ALL,
                           is_alpha: bool = True):
        alpha = self.init_alpha(is_alpha)
        first_series, second_series, third_series = self.init_series(noised_series=noised_series)
        tuples, tuples_noised_first, tuples_noised_second, tuples_noised_third = [], [], [], []

        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            tuples.append(self.stock.create_tuple(self.stock.real_series, i))
            tuples_noised_first.append(self.stock.create_tuple(first_series, i))
            tuples_noised_second.append(self.stock.create_tuple(second_series, i))
            tuples_noised_third.append(self.stock.create_tuple(third_series, i))
        return self.heinrich_relation(tuples_noised_first, tuples, tuple_weights, list(alpha.values())), \
            self.heinrich_relation(tuples_noised_second, tuples, tuple_weights, list(alpha.values())), \
            self.heinrich_relation(tuples_noised_third, tuples, tuple_weights, list(alpha.values()))

    def init_alpha(self, is_alpha: bool):
        return self.custom_alpha if is_alpha else self.default_alpha

    def init_series(self, noised_series: DefectionRange = DefectionRange.ALL):
        series = self.stock.all_series_noised
        if noised_series == DefectionRange.PARTIAL:
            series = self.stock.partially_noised

        first_series, second_series, third_series = series[Strength.WEAK], \
            series[Strength.MODERATE], series[Strength.STRONG]
        return first_series, second_series, third_series

    def draw_heinrich_qualities(self, first: list, second: list, third: list,
                                level: MetricLevel, is_alpha: bool,
                                noise_range: DefectionRange = DefectionRange.ALL, column_name: SeriesColumn = None):
        fig = plt.figure(facecolor="w")
        ax = fig.add_subplot(111, facecolor="#dddddd", axisbelow=True)
        ax.plot(first, "b", lw=1, label=f"Weak noise: std={self.noises_label(Strength.WEAK, noise_range)}")
        ax.plot(second, "r", lw=1, label=f"Medium noise: std={self.noises_label(Strength.MODERATE, noise_range)}")
        ax.plot(third, "g", lw=1, label=f"Strong noise: std={self.noises_label(Strength.STRONG, noise_range)}")
        column = column_name.value if column_name is not None else "all columns"
        noise = noise_range.value if noise_range is not None else "all"
        sensitiveness = ", sensitiveness" if is_alpha else ""
        ax.set_title(f"Heinrich metric {column} prices [{level.value}, {noise} noised{sensitiveness}]")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Quality")
        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.2))
        legend.get_frame().set_alpha(0.5)
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(False)
        plt.show()

    def noises_label(self, strength: Strength, noise_range: DefectionRange):
        if noise_range == DefectionRange.ALL:
            return self.stock.all_noises_strength[strength]
        else:
            return str({column.value: strengths[strength] for column, strengths in
                        self.stock.partially_noised_strength.items()})
