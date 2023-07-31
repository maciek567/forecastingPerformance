import logging
import os
import sys
import warnings
from enum import Enum

import numpy as np
from pandas import read_csv, DataFrame, Series

from inout.intermediate import IntermediateProvider
from inout.paths import pred_stats_csv_path, metrics_scores_csv_path, aggregation_path_metrics, \
    aggregation_path_predictions, aggregation_path_predictions_comparison
from metrics.utils import deviation_range_label, deviation_scale_label, mitigation_label, metric_score_label
from predictions.model import avg_rmse_label, deviations_source_label, deviations_scale_label, \
    deviations_mitigation_label, avg_time_label, std_dev_time_label, avg_mitigation_time_label, avg_mape_label, \
    avg_mae_label, number_label
from timeseries.enums import DeviationSource, sources_short, scales_short, DeviationScale, mitigation_short, Mitigation


class AggregationType(Enum):
    PREDICTION = 0,
    METRIC = 1,
    COMPARISON = 2


class MetricAggregation:

    def aggregate_results(self):
        check_input_existence(metrics_scores_csv_path)
        results = self.collect_data_from_files()
        df = self.calculate_averages(results)
        name = self.determine_name()
        IntermediateProvider.save_as_tex(df, aggregation_path_metrics, name, precision=3)

    @staticmethod
    def collect_data_from_files():
        results = {
            deviation_range_label: [],
            deviation_scale_label: [],
            mitigation_label: [],
            metric_score_label: []
        }

        for file_name in os.listdir(metrics_scores_csv_path):
            if file_name.endswith(".csv"):
                series = read_csv(os.path.join(metrics_scores_csv_path, file_name))

                labels = [deviation_range_label,
                          deviation_scale_label,
                          mitigation_label,
                          metric_score_label]

                for label in labels:
                    results[label].append(series[label])
                print(f"File: {file_name} was included")

        return results

    @staticmethod
    def calculate_averages(results):
        aggregated = {
            number_label: [i for i in range(1, len(results[deviation_range_label][0]) + 1)],
            deviation_range_label: results[deviation_range_label][0],
            deviation_scale_label: results[deviations_scale_label][0],
            mitigation_label: results[deviations_mitigation_label][0]}

        averages = {label: sum(results[label]) / len(results[label]) for label in [metric_score_label]}

        aggregated.update(averages)
        return DataFrame(aggregated)

    @staticmethod
    def determine_name():
        base_name = "average"
        name = base_name
        name_parts = {"company_name": [], "columns": [], "deviation": []}
        for file_name in os.listdir(metrics_scores_csv_path):
            if file_name.endswith(".csv"):
                parts = file_name.split("_")
                name_parts["company_name"].append(parts[0])
                name_parts["columns"].append(parts[1])
                name_parts["deviation"].append(parts[2])

        for occurrences in name_parts.values():
            if occurrences.count(occurrences[0]) == len(occurrences):
                name += f"_{occurrences[0]}"
        return name


class PredictionAggregation:
    avg_time_prepare_label = "Time - prepare data"
    avg_time_model_label = "Time - training model"
    avg_time_prediction_label = "Time - prediction"
    ALL_ROWS_NUMBER = 16

    def aggregate_results(self):
        check_input_existence(pred_stats_csv_path)
        results = self.collect_data_from_files()
        df = self.calculate_averages(results)
        name = self.determine_name()
        IntermediateProvider.save_as_tex(df, aggregation_path_predictions, name, precision=2)

    def collect_data_from_files(self):
        results = {
            self.avg_time_prepare_label: [],
            self.avg_time_model_label: [],
            self.avg_time_prediction_label: [],
            std_dev_time_label: [],
            avg_mitigation_time_label: [],
            avg_rmse_label: [],
            avg_mae_label: [],
            avg_mape_label: []}

        for file_name in os.listdir(pred_stats_csv_path):
            if file_name.endswith(".csv"):
                series = read_csv(os.path.join(pred_stats_csv_path, file_name))
                series[self.avg_time_prepare_label] = self.split_times(series, 0)
                series[self.avg_time_model_label] = self.split_times(series, 1)
                series[self.avg_time_prediction_label] = self.split_times(series, 2)

                labels = [self.avg_time_prepare_label, self.avg_time_model_label, self.avg_time_prediction_label,
                          std_dev_time_label, avg_mitigation_time_label,
                          avg_rmse_label, avg_mae_label, avg_mape_label]

                passed_validation = True
                for label in labels:
                    if len(series[label].dropna()) != self.ALL_ROWS_NUMBER:
                        warnings.warn(
                            f"Skipped file: {file_name}, because it does not contain values for row: {label}.")
                        passed_validation = False
                        break

                if passed_validation:
                    for label in labels:
                        results[label].append(series[label])
                    print(f"File: {file_name} was included")

        return results

    def calculate_averages(self, results):
        none, n = sources_short()[DeviationSource.NONE], sources_short()[DeviationSource.NOISE]
        i, t = sources_short()[DeviationSource.INCOMPLETENESS], sources_short()[DeviationSource.TIMELINESS]
        s, m, h = scales_short()[DeviationScale.SLIGHTLY], scales_short()[DeviationScale.MODERATELY], scales_short()[DeviationScale.HIGHLY]
        no, yes = mitigation_short()[Mitigation.NOT_MITIGATED], mitigation_short()[Mitigation.MITIGATED]
        aggregated = {
            number_label: [i for i in range(1, 17)],
            deviations_source_label: Series(
                [none, n, n, n, n, n, n, i, i, i, i, i, i, t, t, t]),
            deviations_scale_label: Series(
                [none, s, s, m, m, h, h, s, s, m, m, h, h, s, m, h]),
            deviations_mitigation_label: Series(
                [no, no, yes, no, yes, no, yes, no, yes, no, yes, no, yes, no, no, no])}

        averages = {label: sum(results[label]) / len(results[label]) for label in
                    [self.avg_time_prepare_label, self.avg_time_model_label, self.avg_time_prediction_label,
                     std_dev_time_label, avg_mitigation_time_label,
                     avg_rmse_label, avg_mae_label, avg_mape_label]}
        times = {avg_time_label: averages[self.avg_time_prepare_label].round(2).astype(str) + " + " +
                                 averages[self.avg_time_model_label].round(2).astype(str) + " + " +
                                 averages[self.avg_time_prediction_label].round(2).astype(str)}
        aggregated.update(times)
        aggregated.update(averages)

        df = DataFrame(aggregated)
        df = df.drop(columns=[self.avg_time_prepare_label, self.avg_time_model_label, self.avg_time_prediction_label])
        return df

    @staticmethod
    def determine_name():
        base_name = "average"
        name = base_name
        name_parts = {"company_name": [], "column": [], "method": [], "extrapolation_size": []}
        for file_name in os.listdir(pred_stats_csv_path):
            if file_name.endswith(".csv"):
                parts = file_name.split("_")
                name_parts["company_name"].append(parts[0])
                name_parts["column"].append(parts[1])
                name_parts["method"].append(parts[2])
                name_parts["extrapolation_size"].append(parts[3].split(".")[0])

        for occurrences in name_parts.values():
            if occurrences.count(occurrences[0]) == len(occurrences):
                name += f"_{occurrences[0]}"
        return name

    @staticmethod
    def split_times(series, number):
        return Series([[float(time) for time in times.split(" + ")][number] for times in series[avg_time_label]])


class ComparisonAggregation:

    def aggregate_results(self):
        check_input_existence(aggregation_path_predictions_comparison)
        self.show_time_increases()

    @staticmethod
    def show_time_increases():
        for file_name in ["multiple_data_preparation_times", "multiple_prediction_times",
                          "partial_data_preparation_times", "partial_prediction_times", "partial_mape",
                          "performance_data_preparation_times", "performance_model_training_times", "performance_mape"]:
            series = read_csv(os.path.join(aggregation_path_predictions_comparison, file_name) + ".csv")
            reference_key = series.keys().values[0]
            keys_to_compare = series.keys().values[1:]
            df = DataFrame(
                {key: (series[key] - series[reference_key]) / series[reference_key] for key in keys_to_compare})
            min_max = {key: {"min": str(np.argmin(df[key])) + ": " + str(np.round(np.min(df[key] * 100), 1)) + "%",
                             "max": str(np.argmax(df[key])) + ": " + str(np.round(np.max(df[key] * 100), 1)) + "%",
                             "avg": str(np.round(np.mean(df[key] * 100), 1)) + "%"}
                       for key in keys_to_compare}
            print(file_name)
            print(df)
            print(min_max)
            print()

            df_final = DataFrame(min_max).T.reset_index()
            IntermediateProvider.save_as_tex(df_final, aggregation_path_predictions_comparison, file_name, precision=2)


def check_input_existence(path):
    for file_name in os.listdir(path):
        if file_name.endswith('.csv'):
            break
    else:
        logging.exception(f" There are no .csv files in directory: {path}")
        sys.exit(1)
