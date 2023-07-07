import os

import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from numpy import sqrt, mean
from pandas import Series, DataFrame

from timeseries.enums import DeviationSource
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


def prepare_sf_dataframe(data_to_learn, training_size) -> DataFrame:
    series_id = [0 for i in range(0, training_size)]
    return pd.DataFrame({"ds": data_to_learn.keys(), "y": data_to_learn.values, "unique_id": series_id})


def prepare_spark_dataframe(df, spark):
    df['unique_id'] = df['unique_id'].astype(str)
    return spark.createDataFrame(df)


def plot_extrapolation(model, result: ndarray, company_name: str, save_file: bool = False) -> None:
    plt.clf()
    if model.deviation == DeviationSource.NOISE:
        plt.plot(model.data_with_defects, "b", label="Training data", linewidth='1')
        plt.plot(model.actual_data.values, "r", label="Actual data", linewidth='1')
    else:
        plt.plot(model.actual_data.values, "r", label="Actual data", linewidth='1')
        plt.plot(model.data_with_defects, "b", label="Training data", linewidth='1')

    plt.plot(range(model.prediction_start, model.prediction_start + model.predict_size),
             result, "cornflowerblue", label="Extrapolation", linewidth='1')
    plt.axvline(x=model.prediction_border, color='g', label='Prediction start', linestyle="--", linewidth='1')

    method = method_name(model.get_method())
    deviation = f'{model.deviation.value}' + (f', {model.scale.value}' if model.scale is not None else "")
    plt.title(f"{company_name} [{method}, {model.column.value} prices, {deviation}]")
    plt.xlabel(TIME_DAYS_LABEL)
    plt.ylabel(PRICE_USD_LABEL)
    plt.legend()

    if save_file:
        deviation = f'{model.deviation.value}' + (f'_{model.scale.value}' if model.scale is not None else "")
        name = f"{company_name}_{model.column.value}_{method}_{deviation}_{model.predict_size}"
        path = os.path.join('..', 'data', 'predictions', name)
        plt.savefig(f"{path}.pdf", bbox_inches='tight')
    plt.show()
