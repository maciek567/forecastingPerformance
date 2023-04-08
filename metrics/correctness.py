import matplotlib.pyplot as plt

from metrics.utils import MetricLevel
from timeseries.timeseries import StockMarketSeries
from timeseries.utils import SeriesColumn, DefectionRange, DefectsSource, DefectsScale


class HeinrichCorrectnessMetric:

    def __init__(self, stock: StockMarketSeries, alpha: dict = None):
        self.stock = stock
        self.default_alpha = {column: 1.0 for column in SeriesColumn}
        self.custom_alpha = alpha

    @staticmethod
    def d(w_i: float, w_r: float, alpha: float) -> float:
        return pow(abs(w_i - w_r) / max(abs(w_i), abs(w_r)), alpha)

    def heinrich_values(self, w_i: float, w_r: float, alpha: float) -> float:
        return 1 - self.d(w_i, w_r, alpha)

    def heinrich_tuples(self, t: list, e: list, g: list, alpha: list) -> float:
        quality_sum = 0
        for i in range(len(t)):
            quality_sum += self.heinrich_values(t[i], e[i], alpha[i]) * g[i]
        return quality_sum / sum(g)

    def heinrich_relation(self, r: list, e: list, tuple_weights: list, alpha: list) -> float:
        quality_sum = 0
        for i in range(len(r)):
            quality_sum += self.heinrich_tuples(r[i], e[i], tuple_weights, alpha)
        return quality_sum / len(r)

    def values_qualities(self, column: SeriesColumn, is_alpha: bool = True) -> dict:
        alpha = self.get_alpha(is_alpha)
        defected_series = self.stock.get_defected_series(DefectsSource.NOISE)
        qualities = {scale: [] for scale in DefectsScale}

        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            value = self.stock.real_series[column][i]
            for scale in DefectsScale:
                qualities[scale].append(self.heinrich_values(defected_series[scale][column][i], value, alpha[column]))

        return qualities

    def tuples_qualities(self, tuple_weights: list, noise_range: DefectionRange = DefectionRange.ALL,
                         is_alpha: bool = True):
        alpha = self.get_alpha(is_alpha)
        defected_series = self.stock.get_defected_series(DefectsSource.NOISE, noise_range)
        qualities = {scale: [] for scale in DefectsScale}

        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            real_tuple = self.stock.create_tuple(self.stock.real_series, i)
            for scale in DefectsScale:
                qualities[scale].append(self.heinrich_tuples(self.stock.create_tuple(defected_series[scale], i),
                                                             real_tuple, tuple_weights, list(alpha.values())))

        return qualities

    def relation_qualities(self, tuple_weights: list, noise_range: DefectionRange = DefectionRange.ALL,
                           is_alpha: bool = True) -> dict:
        alpha = self.get_alpha(is_alpha)
        defected_series = self.stock.get_defected_series(DefectsSource.NOISE, noise_range)
        real_tuples = []
        defected_tuples = {scale: [] for scale in DefectsScale}

        for i in range(self.stock.time_series_end - self.stock.time_series_start):
            real_tuples.append(self.stock.create_tuple(self.stock.real_series, i))
            for scale in DefectsScale:
                defected_tuples[scale].append(self.stock.create_tuple(defected_series[scale], i))

        return {scale: self.heinrich_relation(defected_tuples[scale], real_tuples, tuple_weights, list(alpha.values()))
                for scale in DefectsScale}

    def get_alpha(self, is_alpha: bool) -> dict:
        return self.custom_alpha if is_alpha else self.default_alpha

    def draw_heinrich_qualities(self, qualities: dict,
                                level: MetricLevel, is_alpha: bool,
                                noise_range: DefectionRange = DefectionRange.ALL,
                                column_name: SeriesColumn = None) -> None:
        fig = plt.figure(facecolor="w")
        ax = fig.add_subplot(111, facecolor="#dddddd", axisbelow=True)
        ax.plot(qualities[DefectsScale.SLIGHTLY], "b", lw=1,
                label=f"Weak noise: std={self.noises_label(DefectsScale.SLIGHTLY, noise_range)}")
        ax.plot(qualities[DefectsScale.MODERATELY], "r", lw=1,
                label=f"Medium noise: std={self.noises_label(DefectsScale.MODERATELY, noise_range)}")
        ax.plot(qualities[DefectsScale.HIGHLY], "g", lw=1,
                label=f"Strong noise: std={self.noises_label(DefectsScale.HIGHLY, noise_range)}")
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

    def noises_label(self, strength: DefectsScale, noise_range: DefectionRange) -> str:
        if noise_range == DefectionRange.ALL:
            return str(self.stock.all_noises_strength[strength])
        else:
            return str({column.value: strengths[strength] for column, strengths in
                        self.stock.partially_noised_strength.items()})
