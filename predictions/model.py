import os
import uuid
import warnings
from statistics import mean, stdev

import pandas as pd
from pandas import DataFrame, concat

from inout.paths import prediction_path
from predictions import utils
from predictions.conditions import do_method_return_extra_params
from predictions.spark import handle_spark
from timeseries.enums import SeriesColumn, sources_short, scales_short, mitigation_short, Mitigation, DeviationScale
from timeseries.timeseries import StockMarketSeries, DeviationRange, DeviationSource

deviations_source_label = "Deviation"
deviations_scale_label = "Scale"
deviations_mitigation_label = "Improve"
avg_time_label = "Time [ms]"
std_dev_time_label = "Time SD"
avg_mitigation_time_label = "M. time"
avg_rmse_label = "RMSE"
avg_mae_label = "MAE"
avg_mape_label = "MAPE"
additional_parameters_label = "(p,q)"


class PredictionModel:

    def __init__(self, stock: StockMarketSeries, prediction_start: int, column: SeriesColumn,
                 deviation_range: DeviationRange = DeviationRange.ALL, deviation_sources: list = None,
                 is_deviation_mitigation: bool = True, deviation_scale: list = None, iterations: int = 5,
                 unique_ids: bool = False):
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
        self.spark = None
        self.unique_ids = unique_ids

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
        self.spark = handle_spark(method)
        self.model_real = self.create_model_real()
        self.model_deviated = self.create_model_deviated_set()
        self.model_mitigated = self.create_model_mitigated_set()
        return self

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

    def plot_prediction(self, source: DeviationSource, scale: DeviationScale = None, mitigation: bool = False,
                        save_file: bool = False) -> None:
        model = None
        if source == DeviationSource.NONE:
            model = self.model_real
        elif not mitigation:
            model = self.model_deviated[source][scale]
        else:
            model = self.model_mitigated[source][scale]
        try:
            prediction_stats = model.extrapolate(self.additional_params)
            to_predict = self.stock.data_size - self.prediction_start
            model.plot_extrapolation(prediction_stats.results, self.stock.company_name, to_predict, save_file=save_file)
        except Exception as e:
            warnings.warn("Prediction method thrown an exception: " + str(e))

    def compute_statistics_set(self, save_file=False) -> None:
        self.compute_statistics(DeviationSource.NONE)
        real = self.compute_statistics(DeviationSource.NONE)
        results = DataFrame([real])

        for deviation_source in self.deviations_sources:
            for deviation_scale in self.deviations_scale:
                deviated = self.compute_statistics(deviation_source, deviation_scale, mitigation=False)
                if deviated:
                    results = concat([results, DataFrame([deviated])], ignore_index=True)
                if self.is_deviation_mitigation and deviation_source in self.deviation_mitigation_sources:
                    mitigated = self.compute_statistics(deviation_source, deviation_scale, mitigation=True)
                    if mitigated:
                        results = concat([results, DataFrame([mitigated])], ignore_index=True)

        self.manage_output(results, save_file)

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
        display = "{:.2f}"
        if results:
            dict_result = {
                deviations_source_label: sources_short()[source],
                deviations_scale_label: scales_short()[scale],
                deviations_mitigation_label: mitigation_short()[mitigation],
                avg_time_label: f"{display.format(mean([r.prepare_time for r in results]))} + {display.format(mean([r.model_time for r in results]))} + {display.format(mean([r.prediction_time for r in results]))}",
                std_dev_time_label: stdev([r.prepare_time + r.model_time + r.prediction_time for r in results]),
                avg_mitigation_time_label: mean([r.mitigation_time for r in results]),
                avg_rmse_label: mean([r.rmse for r in results]),
                avg_mae_label: mean([r.mae for r in results]),
                avg_mape_label: mean(r.mape for r in results)}
            if do_method_return_extra_params(self.method):
                params = results[0].parameters
                p_q = f"({params[0]},{params[1]})"
                dict_result[additional_parameters_label] = p_q
        return dict_result

    def manage_output(self, results: DataFrame, save_file: bool) -> None:
        pd.set_option("display.precision", 2)
        header = \
            f"Statistics [{self.stock.company_name} stock, {self.column.value} price, {self.iterations} iterations]\n"
        text = results.to_string()
        latex = results.to_latex(index=False,
                                 formatters={"name": str.upper},
                                 float_format="{:.2f}".format)
        if not save_file:
            print(header + "\n" + text + "\n\n" + latex)
        else:
            os.makedirs(prediction_path, exist_ok=True)
            values_to_predict = self.stock.data_size - self.prediction_start
            path = f"{prediction_path}{self.stock.company_name}_{self.column.value}_{utils.method_name(self.method)}_{values_to_predict}"
            if self.unique_ids:
                path += f"_{uuid.uuid4()}"
            results.to_csv(path + ".csv")

            latex_file = open(path + ".tex", "w")
            latex_file.write(latex)
            latex_file.close()
