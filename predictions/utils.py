from enum import Enum

from numpy import ndarray, sqrt, mean


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
