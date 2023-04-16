import time
from statistics import mean, stdev

from pandas import DataFrame, Series

from predictions import utils
from timeseries.timeseries import StockMarketSeries, DeviationScale, DeviationRange, DeviationSource
from timeseries.utils import SeriesColumn

deviations_source_label = "Deviations source"
deviations_scale_label = "Deviations scale"
avg_time_label = "Avg time [ms]"
std_dev_time_label = "Std dev time"
avg_rms_label = "Avg RMS"
std_dev_rms_label = "Std dev RMS"


class PredictionModel:

    def __init__(self, stock: StockMarketSeries, prediction_start: int, column: SeriesColumn,
                 deviation_range: DeviationRange = DeviationRange.ALL, deviation_source: DeviationSource = None,
                 deviations_scale: DeviationScale = None, iterations: int = 5):
        self.stock = stock
        self.method = None
        self.prediction_start = prediction_start - stock.time_series_start
        self.column = column
        self.deviation_range = deviation_range
        self.deviations_source = deviation_source if deviation_source is not None \
            else [DeviationSource.NOISE, DeviationSource.INCOMPLETENESS]
        self.deviations_scale = deviations_scale if deviations_scale is not None \
            else [DeviationScale.SLIGHTLY, DeviationScale.MODERATELY, DeviationScale.HIGHLY]
        self.iterations = iterations
        self.additional_params = None
        self.model_real = None
        self.model_deviated = None

    def configure_model(self, method, **kwargs):
        self.method = method
        self.additional_params = kwargs
        self.model_real = self.create_model_real()
        self.model_deviated = self.create_model_deviated_set()
        return self

    def get_series_deviated(self, deviation_range: DeviationRange):
        series_deviated = None
        if deviation_range == DeviationRange.ALL:
            series_deviated = self.stock.all_deviated_series
        elif deviation_range == DeviationRange.PARTIAL:
            series_deviated = self.stock.partially_deviated_series

        return series_deviated

    def create_model_real(self):
        return self.method(self.stock.real_series[self.column], self.prediction_start, self.column,
                           DeviationSource.NONE)

    def create_model_deviated_set(self):
        return {deviation_source: self.create_model_deviated(deviation_source) for deviation_source in
                self.deviations_source}

    def create_model_deviated(self, deviation_source: DeviationSource):
        return {deviation_scale: self.method(
            self.get_series_deviated(self.deviation_range)[deviation_source][deviation_scale][self.column],
            self.prediction_start, self.column,
            self.deviations_source) for deviation_scale in self.deviations_scale}

    def present_prediction(self, source: DeviationSource = None, strength: DeviationScale = None) -> None:
        model = self.model_real
        if source is not None:
            model = self.model_deviated[source][strength]
        result = model.extrapolate(self.additional_params)
        model.plot_extrapolation(result)
        print("RMS: %r " % utils.calculate_rms(model, result))

    def compute_statistics_set(self) -> None:
        results = DataFrame(columns=[deviations_source_label, deviations_scale_label, avg_time_label,
                                     std_dev_time_label, avg_rms_label, std_dev_rms_label])

        result = self.compute_statistics(DeviationSource.NONE)
        results = results.append(result, ignore_index=True)

        for deviation_source in self.deviations_source:
            for deviations_scale in self.deviations_scale:
                result = self.compute_statistics(deviation_source, deviations_scale)
                results = results.append(result, ignore_index=True)

        print(
            f"Statistics [{self.stock.company_name} stock, {self.column.value} price, {self.iterations} iterations]\n")
        print(results)
        print(results.to_latex(index=False,
                               formatters={"name": str.upper},
                               float_format="{:.1f}".format))

    def compute_statistics(self, deviations_source: DeviationSource, deviations_scale: DeviationScale = None) -> dict:
        elapsed_times = []
        rms_metrics = []
        for j in range(self.iterations):
            elapsed_time, rms = self.model_real.extrapolate_and_measure(self.additional_params) \
                if deviations_source is DeviationSource.NONE \
                else self.model_deviated[deviations_source][deviations_scale].extrapolate_and_measure(
                self.additional_params)
            elapsed_times.append(elapsed_time)
            rms_metrics.append(rms)
        return {
            deviations_source_label: "none" if deviations_source is DeviationSource.NONE else deviations_source.value,
            deviations_scale_label: "none" if deviations_scale is None else deviations_scale.value,
            avg_time_label: mean(elapsed_times),
            std_dev_time_label: stdev(elapsed_times),
            avg_rms_label: mean(rms_metrics),
            std_dev_rms_label: stdev(rms_metrics)}


class Prediction:
    def __init__(self, prices: Series, prediction_start: int, column: SeriesColumn, deviation: DeviationSource):
        self.data_to_learn = prices.dropna()[:prediction_start]
        self.data_to_learn_and_validate = prices.dropna()
        self.data_size = len(self.data_to_learn_and_validate)
        self.prediction_start = prediction_start
        self.column = column
        self.deviation = deviation

    def execute_and_measure(self, extrapolation_method, params: dict):
        start_time = time.time_ns()
        extrapolation = extrapolation_method(params)
        elapsed_time = round((time.time_ns() - start_time) / 1e6)
        rms = utils.calculate_rms(self, extrapolation)
        return elapsed_time, rms
