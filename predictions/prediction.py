from statistics import mean, stdev

from pandas import DataFrame

from metrics.utils import Strength, DefectionRange, DefectsSource
from timeseries.timeseries import StockMarketSeries
from timeseries.utils import SeriesColumn


class Prediction:

    def __init__(self, stock: StockMarketSeries, method, prediction_start: int, prediction_end: int,
                 defect_range: DefectionRange, defect_source: DefectsSource, column: SeriesColumn,
                 iterations: int = 5, **kwargs):
        self.stock = stock
        self.method = method
        self.prediction_start = prediction_start
        self.prediction_end = prediction_end
        self.column = column
        self.iterations = iterations
        self.series = stock.series
        self.series_defected = self.get_series_defected(defect_range, defect_source)
        self.model = self.create_model(self.series)
        self.model_defected = self.create_model_defected(self.series_defected)
        self.additional_params = kwargs

    def get_series_defected(self, defect_range, defect_source):
        series_defected = None
        if defect_range == DefectionRange.ALL and defect_source == DefectsSource.NOISE:
            series_defected = self.stock.all_series_noised
        elif defect_range == DefectionRange.ALL and defect_source == DefectsSource.INCOMPLETENESS:
            series_defected = self.stock.all_series_incomplete
        elif defect_range == DefectionRange.PARTIAL and defect_source == DefectsSource.NOISE:
            series_defected = self.stock.partially_noised
        elif defect_range == DefectionRange.PARTIAL and defect_source == DefectsSource.INCOMPLETENESS:
            series_defected = self.stock.partially_incomplete
        return series_defected

    def create_model(self, series):
        return self.method(series[self.column], self.prediction_start, self.prediction_end)

    def create_model_defected(self, series):
        return {defect_scale: self.method(series[defect_scale][self.column], self.prediction_start, self.prediction_end)
                for defect_scale in Strength}

    def present_prediction(self, strength: Strength = None):
        model = self.model
        if strength is not None:
            model = self.model_defected[strength]
        result = model.extrapolate(self.additional_params)
        model.plot_extrapolation(result)
        print("RMS: %r " % model.calculate_rms(result))

    def compute_statistics(self):
        defects_scale_label = "Defects scale"
        avg_time_label = "Avg elapsed time [ms]"
        std_dev_time_label = "Std dev elapsed time"
        avg_rms_label = "Avg RMS"
        std_dev_rms_label = "Std dev RMS"
        results = DataFrame(
            columns=[defects_scale_label, avg_time_label, std_dev_time_label, avg_rms_label, std_dev_rms_label])

        for defects_scale in [None, Strength.WEAK, Strength.MODERATE, Strength.STRONG]:
            elapsed_times = []
            rms_metrics = []
            for j in range(self.iterations):
                elapsed_time, rms = self.model.extrapolate_and_measure(self.additional_params) \
                    if defects_scale is None \
                    else self.model_defected[defects_scale].extrapolate_and_measure(self.additional_params)
                elapsed_times.append(elapsed_time)
                rms_metrics.append(rms)
            results = results.append(
                {defects_scale_label: "none" if defects_scale is None else defects_scale.value,
                 avg_time_label: mean(elapsed_times),
                 std_dev_time_label: stdev(elapsed_times),
                 avg_rms_label: mean(rms_metrics),
                 std_dev_rms_label: stdev(rms_metrics)},
                ignore_index=True)

        print(f"Statistics of {self.column} in {self.stock.company_name} with {self.iterations} iterations:")
        print(results)
