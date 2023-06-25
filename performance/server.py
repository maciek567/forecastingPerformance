import os

from flask import Flask
from pandas import read_csv, DataFrame, Series

from predictions.model import avg_rmse_label, deviations_source_label, deviations_scale_label, \
    deviations_mitigation_label, avg_time_label, std_dev_time_label, avg_mitigation_time_label, avg_mape_label, \
    avg_mae_label

app = Flask(__name__)


@app.route('/prediction', methods=['GET'])
def handle_get_request():
    run_predictions()
    analyze_results()

    response = {'status': 'success'}
    return response


def run_predictions():
    with open("../run/run_predictions.py") as f:
        exec(f.read())


def analyze_results():
    prediction_path = "../data/predictions/"
    results = {
        avg_time_label: [],
        std_dev_time_label: [],
        avg_mitigation_time_label: [],
        avg_rmse_label: [],
        avg_mae_label: [],
        avg_mape_label: []}

    for file_name in os.listdir(prediction_path):
        if file_name.endswith(".csv"):
            series = read_csv(prediction_path + file_name)
            results[avg_time_label].append(series[avg_time_label])
            results[std_dev_time_label].append(series[std_dev_time_label])
            results[avg_mitigation_time_label].append(series[avg_mitigation_time_label])
            results[avg_rmse_label].append(series[avg_rmse_label])
            results[avg_mae_label].append(series[avg_mae_label])
            results[avg_mape_label].append(series[avg_mape_label])

    labels = {
        deviations_source_label: Series(
            ["-", "N", "N", "N", "N", "N", "N", "I", "I", "I", "I", "I", "I", "T", "T", "T"]),
        deviations_scale_label: Series(
            ["-", "S", "S", "M", "M", "H", "H", "S", "S", "M", "M", "H", "H", "S", "M", "H"]),
        deviations_mitigation_label: Series(
            ["N", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N", "Y", "N", "N", "N"])}

    averages = {label: sum(results[label]) / len(results[label]) for label in
                [avg_time_label, std_dev_time_label, avg_mitigation_time_label,
                 avg_rmse_label, avg_mae_label, avg_mape_label]}

    labels.update(averages)

    df = DataFrame(labels)
    latex = df.to_latex(index=False,
                        formatters={"name": str.upper},
                        float_format="{:.2f}".format)
    path = "../data/performance/performance"
    latex_file = open(path + ".tex", "w")
    latex_file.write(latex)
    latex_file.close()


if __name__ == '__main__':
    app.run()
