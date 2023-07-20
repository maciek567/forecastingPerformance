import os

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from numpy import sqrt, mean
from pandas import DataFrame

from inout.paths import pred_graphs_path
from timeseries.enums import DeviationSource, SeriesColumn
from util.graphs import TIME_DAYS_LABEL, PRICE_USD_LABEL


def calculate_rmse(actual: ndarray, result: ndarray) -> float:
    return sqrt(mean((result - actual) ** 2))


def calculate_mae(actual: ndarray, result: ndarray) -> float:
    return mean(abs(result - actual)) * 1.0


def calculate_mape(actual: ndarray, result: ndarray) -> float:
    return mean(abs((result - actual) / actual)) * 100.0


def method_name(method) -> str:
    return str(method)[str(method).index(".") + 1: -2].split(".")[-1]


unique_ids_mapping = {SeriesColumn.OPEN: 0,
                      SeriesColumn.CLOSE: 1,
                      SeriesColumn.ADJ_CLOSE: 2,
                      SeriesColumn.HIGH: 3,
                      SeriesColumn.LOW: 4,
                      SeriesColumn.VOLUME: 5}


def get_column_by_id(column_id):
    return list(unique_ids_mapping.keys())[list(unique_ids_mapping.values()).index(column_id)]


def prepare_sf_dataframe(data_to_learn_dict, training_size) -> DataFrame:
    df = pd.DataFrame(columns=["ds", "y", "unique_id"])
    for column, series in data_to_learn_dict.items():
        series_id = [unique_ids_mapping[column] for i in range(0, training_size[column])]
        df = pd.concat([df, pd.DataFrame({"ds": series.keys(), "y": series.values, "unique_id": series_id})])
    return df


def prepare_spark_dataframe(df, spark):
    df['unique_id'] = df['unique_id'].astype(str)
    return spark.createDataFrame(df)


def extract_predictions(df, results_name) -> dict:
    df = df.reset_index()
    df["unique_id"] = df["unique_id"].astype(int)
    dfs = dict(tuple(df.groupby("unique_id")))
    return {get_column_by_id(column_id): results[results_name] for column_id, results in dfs.items()}


def cut_extrapolation(extrapolation, prediction_delay, columns, data_to_validate):
    return {
        column: extrapolation[column][prediction_delay[column]:prediction_delay[column] + len(data_to_validate[column])]
        for column in columns}


def normalized_columns_weights(columns, weights):
    used_weights = {weight: weights[weight] for weight in weights if weight in columns}
    return {column: weight / sum(used_weights.values()) for column, weight in used_weights.items()}


def plot_extrapolation(model, result: dict, company_name: str, graph_start: int,
                       real_columns: list, deviated_columns: list, save_file: bool = False, shift=0) -> None:
    plt.clf()
    if model.deviation == DeviationSource.NOISE:
        plot_training(plt, model, graph_start)
        plot_actual(plt, model, graph_start)
    else:
        plot_actual(plt, model, graph_start)
        plot_training(plt, model, graph_start)

    plot_results(plt, model, result, graph_start)

    show_titles_and_legend(plt, model, company_name, real_columns, deviated_columns)

    save_to_file(save_file, model, company_name, shift)
    plt.show()


def plot_extrapolations(models: list, results: list, company_name: str, graph_start: int,
                        real_columns: list, deviated_columns: list, save_file: bool = False, shift=0) -> None:
    plt.clf()
    fig, axs = plt.subplots(2, 2)
    axs_dict = {0: axs[0, 0], 1: axs[0, 1], 2: axs[1, 0], 3: axs[1, 1]}
    for i in range(0, len(models)):
        if models[i].deviation == DeviationSource.NOISE:
            plot_training(axs_dict[i], models[i], graph_start)
            plot_actual(axs_dict[i], models[i], graph_start)
        else:
            plot_actual(axs_dict[i], models[i], graph_start)
            plot_training(axs_dict[i], models[i], graph_start)

        plot_results(axs_dict[i], models[i], results[i].results, graph_start)

        show_titles_and_legend(axs_dict[i], models[i], company_name, real_columns[i], deviated_columns[i], fig, i)
    plt.subplots_adjust(left=0.13,
                        bottom=0.13,
                        right=0.83,
                        top=0.83,
                        wspace=0.4,
                        hspace=0.4)

    save_to_file(save_file, models[0], company_name, shift, group=True)
    plt.show()


def plot_actual(axs, model, graph_start):
    colors = ['indigo', 'orangered', 'coral'] if len(model.columns) > 1 else ['orangered']
    i = 0
    for column, series in sort_dict(model.actual_data).items():
        axs.plot(series.values[graph_start:], "r", label=f"Actual: {column.value}", linewidth='0.7',
                 color=colors[i % len(colors)])
        i += 1


def plot_training(axs, model, graph_start):
    colors = ['royalblue', 'darkorange', 'navy', 'darkviolet'] if len(model.columns) > 1 else ['royalblue']
    i = 0
    for column, series in sort_dict(model.data_with_defects).items():
        axs.plot(series[graph_start:], "b", label=f"Training: {column.value}", linewidth='0.7',
                 color=colors[i % len(colors)])
        i += 1


def plot_results(axs, model, result, graph_start):
    colors = ['cornflowerblue', 'orange', 'blue', 'violet'] if len(model.columns) > 1 else ['forestgreen']
    i = 0
    for column, series in sort_dict(result).items():
        prediction_start = model.prediction_start
        axs.plot(range(prediction_start - graph_start, prediction_start + model.validation_size - graph_start),
                 series, label=f"Extrapolation: {column.value}", linewidth='1.0', color=colors[i % len(colors)])
        i += 1
    axs.axvline(x=min_prediction_start(model.training_end) - graph_start, color='g', label='Prediction start',
                linestyle="--", linewidth='1')


def show_titles_and_legend(graph, model, company_name, real_columns, deviated_columns, fig=None, i=None):
    real_columns, deviated_columns = column_values(real_columns), column_values(deviated_columns)
    method = method_name(model.get_method())
    case_letter = {0: "a)", 1: "b)", 2: "c)", 3: "d)"}

    plt.suptitle(f"{company_name} stock extrapolation: {method}", fontsize=14)
    letter = f"{case_letter[i]}" if i is not None else ""
    deviation_subtitle = f'{letter} not deviated: {real_columns}' if len(real_columns) > 0 else f"{letter} "
    if model.deviation != DeviationSource.NONE and len(deviated_columns) > 0:
        deviation_subtitle = deviation_subtitle + ", " if len(deviation_subtitle) > 3 else deviation_subtitle
        deviation_subtitle = deviation_subtitle + f"{model.scale.value} {model.deviation.value}: {deviated_columns}"
    if type(graph) == type(matplotlib):
        graph.title(deviation_subtitle, fontsize=10)
        graph.xlabel(TIME_DAYS_LABEL)
        graph.ylabel(PRICE_USD_LABEL)
    else:
        graph.set_title(deviation_subtitle, fontsize=10)
        fig.supxlabel(TIME_DAYS_LABEL)
        fig.supylabel(PRICE_USD_LABEL)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")


def save_to_file(save_file, model, company_name, shift, group=False):
    method = method_name(model.get_method())
    if save_file:
        os.makedirs(pred_graphs_path, exist_ok=True)
        deviation = f'{model.deviation.value}' + (f'_{model.scale.value}' if model.scale is not None else "")
        name = f"{company_name}_{'_'.join([column.value for column in model.columns])}_{method}"
        name += f"_{deviation}_{model.validation_size}" if not group else f"_group_{model.validation_size}"
        if shift != 0:
            name += f"_{shift}"
        path = os.path.join(pred_graphs_path, name)
        plt.savefig(f"{path}.pdf", bbox_inches='tight')


def min_prediction_start(prediction_start_dict):
    return min([pred_start for pred_start in prediction_start_dict.values()])


def sort_dict(dict_to_sort) -> dict:
    return dict(sorted(dict_to_sort.items(), key=lambda x: x[0].value))


def column_values(columns):
    return [column.value for column in columns]
