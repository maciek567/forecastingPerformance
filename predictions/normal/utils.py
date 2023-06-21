from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import Series, DataFrame

from predictions.utils import PredictionMethod
from util.graphs import TIME_DAYS_LABEL, PRICE_USD_LABEL


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
