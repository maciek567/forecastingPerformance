import os

import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from numpy import sqrt, mean
from pandas import Series, DataFrame

from timeseries.enums import DeviationSource, SeriesColumn
from util.graphs import TIME_DAYS_LABEL, PRICE_USD_LABEL


def normalize(series: Series, original_series: Series) -> DataFrame:
    df = DataFrame((series - original_series.min()) / (original_series.max() - original_series.min()))
    df.columns = ["y"]
    return df


def denormalize(series: ndarray, original_series: Series) -> ndarray:
    return series * (original_series.max() - original_series.min()) + original_series.min()


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
    dfs = dict(tuple(df.groupby("unique_id")))
    return {get_column_by_id(column_id): results[results_name] for column_id, results in dfs.items()}


def normalized_columns_weights(columns, weights):
    used_weights = {weight: weights[weight] for weight in weights if weight in columns}
    return {column: weight / sum(used_weights.values()) for column, weight in used_weights.items()}


def plot_extrapolation(model, result: dict, company_name: str, save_file: bool = False) -> None:
    plt.clf()
    if model.deviation == DeviationSource.NOISE:
        plot_defects(model)
        plot_actual(model)
    else:
        plot_actual(model)
        plot_defects(model)

    plot_results(model, result)
    plt.axvline(x=model.prediction_border, color='g', label='Prediction start', linestyle="--", linewidth='1')

    method = method_name(model.get_method())
    deviation = f'{model.deviation.value}' + (f', {model.scale.value}' if model.scale is not None else "")
    plt.title(f"{company_name} [{method}, {deviation}]")
    plt.xlabel(TIME_DAYS_LABEL)
    plt.ylabel(PRICE_USD_LABEL)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    if save_file:
        deviation = f'{model.deviation.value}' + (f'_{model.scale.value}' if model.scale is not None else "")
        name = f"{company_name}_{'_'.join([column.value for column in model.columns])}_{method}_{deviation}_{model.predict_size}"
        path = os.path.join('..', 'data', 'predictions', name)
        plt.savefig(f"{path}.pdf", bbox_inches='tight')
    plt.show()


def plot_actual(model):
    colors = ['indigo', 'orangered', 'coral'] if len(model.columns) > 1 else ['orangered']
    i = 0
    for column, series in sort_dict(model.actual_data).items():
        plt.plot(series.values, "r", label=f"Actual: {column.value}", linewidth='0.7', color=colors[i % len(colors)])
        i += 1


def plot_defects(model):
    colors = ['royalblue', 'darkorange', 'navy', 'darkviolet'] if len(model.columns) > 1 else ['royalblue']
    i = 0
    for column, series in sort_dict(model.data_with_defects).items():
        plt.plot(series, "b", label=f"Training: {column.value}", linewidth='0.7', color=colors[i % len(colors)])
        i += 1


def plot_results(model, result):
    colors = ['cornflowerblue', 'orange', 'blue', 'violet'] if len(model.columns) > 1 else ['forestgreen']
    i = 0
    for column, series in sort_dict(result).items():
        plt.plot(range(model.prediction_start, model.prediction_start + model.predict_size),
                 series, label=f"Extrapolation: {column.value}", linewidth='1.0', color=colors[i % len(colors)])
        i += 1


def sort_dict(dict_to_sort) -> dict:
    return dict(sorted(dict_to_sort.items(), key=lambda x: x[0].value))
