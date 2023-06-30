import os

from matplotlib import pyplot as plt
from numpy import ndarray
from numpy import sqrt, mean
from pandas import Series, DataFrame

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


def plot_extrapolation(model, result: ndarray, method, company_name, to_predict, save_file: bool = False) -> None:
    plt.plot(model.data_to_learn_and_validate.values, "r", label="Actual data")
    plt.plot(range(model.prediction_start, model.data_size), result, "b", label="Prediction")
    plt.axvline(x=model.prediction_start, color='g', label='Extrapolation start')
    plt.title(f"{company_name} [{method_name(method)}, {model.column.value} prices, {model.deviation.value}]")
    plt.xlabel(TIME_DAYS_LABEL)
    plt.ylabel(PRICE_USD_LABEL)
    plt.legend()

    if save_file:
        name = f"{company_name}_{model.column.value}_{method_name(method)}_{model.deviation.value}_{to_predict}"
        path = os.path.join('..', 'data', 'predictions', name)
        plt.savefig(f"{path}.pdf", bbox_inches='tight')
    plt.show()
