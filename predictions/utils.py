from enum import Enum

from matplotlib import pyplot as plt
from numpy import ndarray, sqrt, mean
from pandas import Series, DataFrame

from util.graphs import TIME_DAYS_LABEL, PRICE_USD_LABEL


class PredictionMethod(Enum):
    Arima = "Arima"
    XGBoost = "XGBoost"
    Reservoir = "Reservoir computing"


def calculate_rmse(actual: ndarray, result: ndarray) -> float:
    return sqrt(mean((result - actual) ** 2))


def calculate_mae(actual: ndarray, result: ndarray) -> float:
    return mean(abs(result - actual)) * 1.0


def calculate_mape(actual: ndarray, result: ndarray) -> float:
    return mean(abs((result - actual) / actual)) * 100.0


def normalize(series: Series, original_series: Series) -> DataFrame:
    df = DataFrame((series - original_series.min()) / (original_series.max() - original_series.min()))
    df.columns = ["y"]
    return df


def denormalize(series: ndarray, original_series: Series) -> ndarray:
    return series * (original_series.max() - original_series.min()) + original_series.min()


def plot_extrapolation(model, result: ndarray, method: PredictionMethod) -> None:
    plt.plot(model.data_to_learn_and_validate.values, "r", label="Actual data")
    plt.plot(range(model.prediction_start, model.data_size), result, "b", label="Prediction")
    plt.axvline(x=model.prediction_start, color='g', label='Extrapolation start')
    plt.title(f"{method.value} extrapolation [{model.column.value} prices, {model.deviation.value}]")
    plt.xlabel(TIME_DAYS_LABEL)
    plt.ylabel(PRICE_USD_LABEL)
    plt.legend()
    plt.show()
