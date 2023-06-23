import os
import warnings
from statistics import mean, stdev

import pandas as pd
from pandas import DataFrame, concat
from pyspark.sql import SparkSession

from timeseries.enums import SeriesColumn, sources_short, scales_short, mitigation_short, Mitigation, DeviationScale
from timeseries.timeseries import StockMarketSeries, DeviationRange, DeviationSource

deviations_source_label = "Deviation"
deviations_scale_label = "Scale"
deviations_mitigation_label = "Mitigation"
avg_time_label = "Time [ms]"
std_dev_time_label = "Time SD"
avg_mitigation_time_label = "M. time"
avg_rmse_label = "RMSE"
avg_mae_label = "MAE"
avg_mape_label = "MAPE"
std_dev_mape_label = "MAPE SD"


class PredictionModel:

    def __init__(self, stock: StockMarketSeries, prediction_start: int, column: SeriesColumn,
                 deviation_range: DeviationRange = DeviationRange.ALL, deviation_sources: list = None,
                 is_deviation_mitigation: bool = True, deviation_scale: list = None, iterations: int = 5,
                 is_spark: bool = False):
        self.stock = stock
        self.prediction_start = prediction_start
        self.column = column
        self.deviation_range = deviation_range
        self.deviations_sources = deviation_sources if deviation_sources is not None \
            else [DeviationSource.NOISE, DeviationSource.INCOMPLETENESS, DeviationSource.TIMELINESS]
        self.is_deviation_mitigation = is_deviation_mitigation
        self.deviation_mitigation_sources = self.get_deviation_mitigation_sources()
        self.deviations_scale = deviation_scale if deviation_scale is not None \
            else [DeviationScale.SLIGHTLY, DeviationScale.MODERATELY, DeviationScale.HIGHLY]
        self.iterations = iterations
        self.method = None
        self.additional_params = None
        self.model_real = None
        self.model_deviated = None
        self.model_mitigated = None
        self.spark = self.start_spark() if is_spark else None

    def get_deviation_mitigation_sources(self) -> list:
        if self.is_deviation_mitigation:
            mitigation_sources = self.deviations_sources.copy()
            if DeviationSource.TIMELINESS in mitigation_sources:
                mitigation_sources.remove(DeviationSource.TIMELINESS)
            return mitigation_sources
        else:
            return []

    def configure_model(self, method, **kwargs):
        self.method = method
        self.additional_params = kwargs
        self.model_real = self.create_model_real()
        self.model_deviated = self.create_model_deviated_set()
        self.model_mitigated = self.create_model_mitigated_set()
        return self

    @staticmethod
    def start_spark():
        return SparkSession.builder.appName("Predictions").getOrCreate()

    def create_model_real(self):
        return self.method(prices=self.stock.real_series[self.column],
                           real_prices=self.stock.real_series[self.column],
                           prediction_border=self.prediction_start,
                           prediction_delay=0,
                           column=self.column,
                           deviation=DeviationSource.NONE,
                           spark=self.spark)

    def create_model_deviated_set(self):
        return {deviation_source: self.create_model_deviated(deviation_source) for deviation_source in
                self.deviations_sources}

    def create_model_mitigated_set(self):
        return {deviation_source: self.create_model_mitigated(deviation_source) for deviation_source in
                self.deviation_mitigation_sources}

    def create_model_deviated(self, source: DeviationSource):
        return {deviation_scale:
            self.method(
                prices=self.get_series_deviated(self.deviation_range)[source][deviation_scale][self.column],
                real_prices=self.stock.real_series[self.column] if source is not DeviationSource.TIMELINESS else
                self.get_series_deviated(self.deviation_range)[source][deviation_scale][self.column],
                prediction_border=self.prediction_start,
                prediction_delay=self.stock.obsolescence.obsolescence_scale[
                    deviation_scale] if source == DeviationSource.TIMELINESS else 0,
                column=self.column,
                deviation=source,
                spark=self.spark)
            for deviation_scale in self.deviations_scale}

    def create_model_mitigated(self, source: DeviationSource):
        return {deviation_scale:
            self.method(
                prices=self.stock.mitigated_deviations_series[source][deviation_scale][self.column][Mitigation.DATA],
                mitigation_time=self.stock.mitigated_deviations_series[source][deviation_scale][self.column][
                    Mitigation.TIME],
                real_prices=self.stock.real_series[self.column],
                prediction_border=self.prediction_start,
                prediction_delay=0,
                column=self.column,
                deviation=source,
                spark=self.spark)
            for deviation_scale in self.deviations_scale}

    def get_series_deviated(self, deviation_range: DeviationRange):
        series_deviated = None
        if deviation_range == DeviationRange.ALL:
            series_deviated = self.stock.all_deviated_series
        elif deviation_range == DeviationRange.PARTIAL:
            series_deviated = self.stock.partially_deviated_series

        return series_deviated

    def plot_prediction(self, source: DeviationSource, scale: DeviationScale = None) -> None:
        model = self.model_real if source == DeviationSource.NONE else self.model_deviated[source][scale]
        extrapolation = model.extrapolate(self.additional_params)
        model.plot_extrapolation(extrapolation)

    def compute_statistics_set(self, save_to_file=False) -> None:
        results = DataFrame(columns=[deviations_source_label, deviations_scale_label, deviations_mitigation_label,
                                     avg_time_label, std_dev_time_label, avg_mitigation_time_label,
                                     avg_rmse_label, avg_mae_label, avg_mape_label, std_dev_mape_label])

        real = self.compute_statistics(DeviationSource.NONE)
        results = results.append(real, ignore_index=True)

        for deviation_source in self.deviations_sources:
            for deviation_scale in self.deviations_scale:
                deviated = self.compute_statistics(deviation_source, deviation_scale, mitigation=False)
                if deviated:
                    results = concat([results, DataFrame([deviated])], ignore_index=True)
                if self.is_deviation_mitigation and deviation_source in self.deviation_mitigation_sources:
                    mitigated = self.compute_statistics(deviation_source, deviation_scale, mitigation=True)
                    if mitigated:
                        results = concat([results, DataFrame([mitigated])], ignore_index=True)

        self.manage_output(results, save_to_file)

    def compute_statistics(self, source: DeviationSource, scale: DeviationScale = None,
                           mitigation: bool = False) -> dict:
        results = []

        for j in range(self.iterations):
            result = None
            try:
                if source is DeviationSource.NONE:
                    result = self.model_real.extrapolate_and_measure(self.additional_params)
                elif not mitigation:
                    result = self.model_deviated[source][scale].extrapolate_and_measure(self.additional_params)
                else:
                    result = self.model_mitigated[source][scale].extrapolate_and_measure(self.additional_params)
                    result.mitigation_time = self.model_mitigated[source][scale].mitigation_time
                results.append(result)
            except Exception as e:
                warnings.warn("Prediction method thrown an exception: " + str(e))

        dict_result = {}
        if results:
            dict_result = {
                deviations_source_label: sources_short()[source],
                deviations_scale_label: scales_short()[scale],
                deviations_mitigation_label: mitigation_short()[mitigation],
                avg_time_label: mean([r.elapsed_time for r in results]),
                std_dev_time_label: stdev([r.elapsed_time for r in results]),
                avg_mitigation_time_label: mean([r.mitigation_time for r in results]),
                avg_rmse_label: mean([r.rmse for r in results]),
                avg_mae_label: mean([r.mae for r in results]),
                avg_mape_label: mean(r.mape for r in results),
                std_dev_mape_label: stdev([r.mape for r in results])}
        return dict_result

    def manage_output(self, results: DataFrame, save_to_file: bool) -> None:
        pd.set_option("display.precision", 2)
        header = \
            f"Statistics [{self.stock.company_name} stock, {self.column.value} price, {self.iterations} iterations]\n"
        text = results.to_string()
        latex = results.to_latex(index=False,
                                 formatters={"name": str.upper},
                                 float_format="{:.2f}".format)
        if not save_to_file:
            print(header + "\n" + text + "\n\n" + latex)
        else:
            base_path = "../data/predictions"
            os.makedirs(base_path, exist_ok=True)
            path = f"{base_path}/{self.stock.company_name}_{self.column.value}_{self.method_name()}"
            results.to_csv(path + ".csv")

            latex_file = open(path + ".tex", "w")
            latex_file.write(latex)
            latex_file.close()

    def method_name(self) -> str:
        return str(self.method)[str(self.method).index(".") + 1: -2].split(".")[-1]

