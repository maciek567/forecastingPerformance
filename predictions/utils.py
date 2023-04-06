from enum import Enum

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import Series, DataFrame


class PredictionMethod(Enum):
    Arima = "Arima"
    XGBoost = "XGBoost"
    Reservoir = "Reservoir computing"


def calculate_rms(model, result: ndarray) -> float:
    actual = model.data_to_learn_and_validate[model.prediction_start:].values
    return round(np.sqrt(np.mean((result - actual) ** 2)), 3)


def normalize(series: Series) -> DataFrame:
    df = pd.DataFrame((series - series.min()) / (series.max() - series.min()))
    df.columns = ["y"]
    return df


def denormalize(series: Series, original_series: Series) -> Series:
    return series * (original_series.max() - original_series.min()) + original_series.min()


def plot_extrapolation(model, result: ndarray, method: PredictionMethod) -> None:
    plt.plot(model.data_to_learn_and_validate.values, "r", label="Actual data")
    plt.plot(range(model.prediction_start, model.data_size), result, "b",
             label="Prediction")
    plt.axvline(x=model.prediction_start, color='g', label='Extrapolation start')
    plt.title(f"{method.value} extrapolation [{model.column.value} prices, {model.defect.value}]")
    plt.xlabel("Time [days]")
    plt.ylabel("Prices [USD]")
    plt.legend()
    plt.show()
