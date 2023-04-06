from statistics import mean, stdev

from pandas import DataFrame

from metrics.utils import DefectsScale, DefectionRange, DefectsSource
from predictions import utils
from timeseries.timeseries import StockMarketSeries
from timeseries.utils import SeriesColumn

defects_source_label = "Defects source"
defects_scale_label = "Defects scale"
avg_time_label = "Avg elapsed time [ms]"
std_dev_time_label = "Std dev elapsed time"
avg_rms_label = "Avg RMS"
std_dev_rms_label = "Std dev RMS"


class PredictionModel:

    def __init__(self, stock: StockMarketSeries, prediction_start: int, column: SeriesColumn,
                 defect_range: DefectionRange = DefectionRange.ALL, defect_source: DefectsSource = None,
                 defects_scale: DefectsScale = None, iterations: int = 5):
        self.stock = stock
        self.method = None
        self.prediction_start = prediction_start - stock.time_series_start
        self.column = column
        self.defect_range = defect_range
        self.defects_source = defect_source if defect_source is not None \
            else [DefectsSource.NOISE, DefectsSource.INCOMPLETENESS]
        self.defects_scale = defects_scale if defects_scale is not None \
            else [DefectsScale.SLIGHTLY, DefectsScale.MODERATELY, DefectsScale.HIGHLY]
        self.iterations = iterations
        self.additional_params = None
        self.model_real = None
        self.model_defected = None

    def create_model(self, method, **kwargs):
        self.method = method
        self.additional_params = kwargs
        self.model_real = self.create_model_real()
        self.model_defected = self.create_model_defected_set()
        return self

    def get_series_defected(self, defect_range: DefectionRange):
        series_defected = None
        if defect_range == DefectionRange.ALL:
            series_defected = self.stock.all_defected_series
        elif defect_range == DefectionRange.PARTIAL:
            series_defected = self.stock.partially_defected_series

        return series_defected

    def create_model_real(self):
        return self.method(self.stock.real_series[self.column], self.prediction_start, self.column, DefectsSource.NONE)

    def create_model_defected_set(self):
        return {defect_source: self.create_model_defected(defect_source) for defect_source in self.defects_source}

    def create_model_defected(self, defect_source: DefectsSource):
        return {defect_scale: self.method(
            self.get_series_defected(self.defect_range)[defect_source][defect_scale][self.column],
            self.prediction_start, self.column,
            self.defects_source) for defect_scale in self.defects_scale}

    def present_prediction(self, source: DefectsSource = None, strength: DefectsScale = None) -> None:
        model = self.model_real
        if source is not None:
            model = self.model_defected[source][strength]
        result = model.extrapolate(self.additional_params)
        model.plot_extrapolation(result)
        print("RMS: %r " % utils.calculate_rms(model, result))

    def compute_statistics_set(self) -> None:
        results = DataFrame(columns=[defects_source_label, defects_scale_label, avg_time_label,
                                     std_dev_time_label, avg_rms_label, std_dev_rms_label])

        result = self.compute_statistics(DefectsSource.NONE)
        results = results.append(result, ignore_index=True)

        for defect_source in self.defects_source:
            for defects_scale in self.defects_scale:
                result = self.compute_statistics(defect_source, defects_scale)
                results = results.append(result, ignore_index=True)

        print(
            f"Statistics [{self.stock.company_name} stock, {self.column.value} price, {self.iterations} iterations]\n")
        print(results)

    def compute_statistics(self, defects_source: DefectsSource, defects_scale: DefectsScale = None) -> dict:
        elapsed_times = []
        rms_metrics = []
        for j in range(self.iterations):
            elapsed_time, rms = self.model_real.extrapolate_and_measure(self.additional_params) \
                if defects_source is DefectsSource.NONE \
                else self.model_defected[defects_source][defects_scale].extrapolate_and_measure(self.additional_params)
            elapsed_times.append(elapsed_time)
            rms_metrics.append(rms)
        return {defects_source_label: "none" if defects_source is DefectsSource.NONE else defects_source.value,
                defects_scale_label: "none" if defects_scale is None else defects_scale.value,
                avg_time_label: mean(elapsed_times),
                std_dev_time_label: stdev(elapsed_times),
                avg_rms_label: mean(rms_metrics),
                std_dev_rms_label: stdev(rms_metrics)}
