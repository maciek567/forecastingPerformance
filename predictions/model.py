from statistics import mean, stdev

from pandas import DataFrame

from timeseries.timeseries import StockMarketSeries, DeviationScale, DeviationRange, DeviationSource
from timeseries.utils import SeriesColumn, sources_short, scales_short

deviations_source_label = "Deviation"
deviations_scale_label = "Scale"
avg_time_label = "Time"
std_dev_time_label = "Time SD"
avg_rmse_label = "RMSE"
avg_mae_label = "MAE"
avg_mape_label = "MAPE"
std_dev_mape_label = "MAPE SD"


class PredictionModel:

    def __init__(self, stock: StockMarketSeries, prediction_start: int, column: SeriesColumn,
                 deviation_range: DeviationRange = DeviationRange.ALL, deviation_source: DeviationSource = None,
                 deviations_scale: DeviationScale = None, iterations: int = 5):
        self.stock = stock
        self.prediction_start = prediction_start - stock.time_series_start
        self.column = column
        self.deviation_range = deviation_range
        self.deviations_source = deviation_source if deviation_source is not None \
            else [DeviationSource.NOISE, DeviationSource.INCOMPLETENESS, DeviationSource.TIMELINESS]
        self.deviations_scale = deviations_scale if deviations_scale is not None \
            else [DeviationScale.SLIGHTLY, DeviationScale.MODERATELY, DeviationScale.HIGHLY]
        self.iterations = iterations
        self.method = None
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
        return self.method(prices=self.stock.real_series[self.column],
                           real_prices=self.stock.real_series[self.column],
                           training_set_end=self.prediction_start,
                           prediction_delay=0,
                           column=self.column,
                           deviation=DeviationSource.NONE)

    def create_model_deviated_set(self):
        return {deviation_source: self.create_model_deviated(deviation_source) for deviation_source in
                self.deviations_source}

    def create_model_deviated(self, source: DeviationSource):
        return {deviation_scale:
            self.method(
                prices=self.get_series_deviated(self.deviation_range)[source][deviation_scale][self.column],
                real_prices=self.stock.real_series[self.column],
                training_set_end=self.prediction_start,
                prediction_delay=self.stock.obsolescence.obsolescence_scale[
                    deviation_scale] if source == DeviationSource.TIMELINESS else 0,
                column=self.column,
                deviation=source)
            for deviation_scale in self.deviations_scale}

    def plot_prediction(self, source: DeviationSource, scale: DeviationScale = None) -> None:
        model = self.model_real if source == DeviationSource.NONE else self.model_deviated[source][scale]
        extrapolation = model.extrapolate(self.additional_params)
        model.plot_extrapolation(extrapolation)

    def compute_statistics_set(self) -> None:
        results = DataFrame(columns=[deviations_source_label, deviations_scale_label,
                                     avg_time_label, std_dev_time_label,
                                     avg_rmse_label, avg_mae_label, avg_mape_label, std_dev_mape_label])

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

    def compute_statistics(self, source: DeviationSource, scale: DeviationScale = None) -> dict:
        results = []
        for j in range(self.iterations):
            prediction_results = self.model_real.extrapolate_and_measure(self.additional_params) \
                if source is DeviationSource.NONE \
                else self.model_deviated[source][scale].extrapolate_and_measure(self.additional_params)
            results.append(prediction_results)

        return {
            deviations_source_label: sources_short()[source],
            deviations_scale_label: scales_short()[scale],
            avg_time_label: mean([r.elapsed_time for r in results]),
            std_dev_time_label: stdev([r.elapsed_time for r in results]),
            avg_rmse_label: mean([r.rmse for r in results]),
            avg_mae_label: mean([r.mae for r in results]),
            avg_mape_label: mean(r.mape for r in results),
            std_dev_mape_label: stdev([r.mape for r in results])}
