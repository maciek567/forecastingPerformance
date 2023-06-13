from enum import Enum

from numpy import ndarray, sqrt, mean
from pyspark.sql import DataFrame
from pyspark.sql.functions import min, max


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


def normalize(series: DataFrame, original_series: DataFrame) -> DataFrame:
    min_val = original_series.select(min("Values")).collect()
    max_val = original_series.select(max("Values")).collect()
    df = (series - min_val) / (max_val - min_val)
    df.columns = ["y"]
    return df


def denormalize(series: ndarray, original_series: DataFrame) -> ndarray:
    return series * (original_series.max() - original_series.min()) + original_series.min()
