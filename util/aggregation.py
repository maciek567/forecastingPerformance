import os
import warnings
from enum import Enum

from pandas import read_csv, DataFrame, Series

from inout.intermediate import IntermediateProvider
from inout.paths import pred_stats_csv_path, metrics_scores_csv_path, aggregation_path_metrics, \
    aggregation_path_predictions
from metrics.utils import deviation_range_label, deviation_scale_label, mitigation_label, metric_score_label
from predictions.model import avg_rmse_label, deviations_source_label, deviations_scale_label, \
    deviations_mitigation_label, avg_time_label, std_dev_time_label, avg_mitigation_time_label, avg_mape_label, \
    avg_mae_label, number_label


class AggregationType(Enum):
    PREDICTION = 0,
    METRIC = 1


class MetricAggregation:

    def aggregate_results(self):
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
        aggregated = {
            number_label: [i for i in range(1, 17)],
            deviations_source_label: Series(
                ["-", "N", "N", "N", "N", "N", "N", "I", "I", "I", "I", "I", "I", "T", "T", "T"]),
            deviations_scale_label: Series(
                ["-", "S", "S", "M", "M", "H", "H", "S", "S", "M", "M", "H", "H", "S", "M", "H"]),
            deviations_mitigation_label: Series(
                ["N", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N", "N", "N"])}

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
