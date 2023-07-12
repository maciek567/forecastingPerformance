import os

from pandas import read_csv, DataFrame, Series

from inout.paths import aggregation_path, pred_stats_csv_path
from predictions.model import avg_rmse_label, deviations_source_label, deviations_scale_label, \
    deviations_mitigation_label, avg_time_label, std_dev_time_label, avg_mitigation_time_label, avg_mape_label, \
    avg_mae_label, number_label

avg_time_prepare_label = "Time - prepare data"
avg_time_model_label = "Time - training model"
avg_time_prediction_label = "Time - prediction"


def aggregate_results():
    results = collect_data_from_files()
    df = calculate_averages(results)
    name = determine_name()
    save_aggregation_to_file(df, name)


def split_times(series, number):
    return Series([[float(time) for time in times.split(" + ")][number] for times in series[avg_time_label]])


def collect_data_from_files():
    results = {
        avg_time_prepare_label: [],
        avg_time_model_label: [],
        avg_time_prediction_label: [],
        std_dev_time_label: [],
        avg_mitigation_time_label: [],
        avg_rmse_label: [],
        avg_mae_label: [],
        avg_mape_label: []}

    for file_name in os.listdir(pred_stats_csv_path):
        series = read_csv(os.path.join(pred_stats_csv_path, file_name))
        results[avg_time_prepare_label].append(split_times(series, 0))
        results[avg_time_model_label].append(split_times(series, 1))
        results[avg_time_prediction_label].append(split_times(series, 2))
        results[std_dev_time_label].append(series[std_dev_time_label])
        results[avg_mitigation_time_label].append(series[avg_mitigation_time_label])
        results[avg_rmse_label].append(series[avg_rmse_label])
        results[avg_mae_label].append(series[avg_mae_label])
        results[avg_mape_label].append(series[avg_mape_label])

    return results


def calculate_averages(results):
    aggregated = {
        number_label: [i for i in range(1, 17)],
        deviations_source_label: Series(
            ["-", "N", "N", "N", "N", "N", "N", "I", "I", "I", "I", "I", "I", "T", "T", "T"]),
        deviations_scale_label: Series(
            ["-", "S", "S", "M", "M", "H", "H", "S", "S", "M", "M", "H", "H", "S", "M", "H"]),
        deviations_mitigation_label: Series(
            ["N", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N", "N", "N"])}

    averages = {label: sum(results[label]) / len(results[label]) for label in
                [avg_time_prepare_label, avg_time_model_label, avg_time_prediction_label,
                 std_dev_time_label, avg_mitigation_time_label,
                 avg_rmse_label, avg_mae_label, avg_mape_label]}
    times = {avg_time_label: averages[avg_time_prepare_label].round(2).astype(str) + " + " +
                             averages[avg_time_model_label].round(2).astype(str) + " + " +
                             averages[avg_time_prediction_label].round(2).astype(str)}
    aggregated.update(times)
    aggregated.update(averages)

    df = DataFrame(aggregated)
    df = df.drop(columns=[avg_time_prepare_label, avg_time_model_label, avg_time_prediction_label])
    return df


def determine_name():
    base_name = "average"
    name = base_name
    name_parts = {"company_name": [], "column": [], "method": [], "extrapolation_size": []}
    for file_name in os.listdir(pred_stats_csv_path):
        parts = file_name.split("_")
        name_parts["company_name"].append(parts[0])
        name_parts["column"].append(parts[1])
        name_parts["method"].append(parts[2])
        name_parts["extrapolation_size"].append(parts[3].split(".")[0])

    for occurrences in name_parts.values():
        if occurrences.count(occurrences[0]) == len(occurrences):
            name += f"_{occurrences[0]}"
    name += ".tex"
    return name


def save_aggregation_to_file(df, name):
    os.makedirs(aggregation_path, exist_ok=True)
    latex = df.to_latex(index=False,
                        formatters={"name": str.upper},
                        float_format="{:.2f}".format)

    latex_file = open(os.path.join(aggregation_path, name), "w")
    latex_file.write(latex)
    latex_file.close()


if __name__ == '__main__':
    aggregate_results()
    print("DONE")
