import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from predictions import utils
from predictions.prediction import Prediction
from predictions.utils import PredictionMethod
from timeseries.utils import SeriesColumn, DeviationSource


class ArimaPrediction(Prediction):
    def __init__(self, prices: Series, prediction_start: int, column: SeriesColumn, deviation: DeviationSource):
        super().__init__(prices, prediction_start, column, deviation)

    @staticmethod
    def print_elapsed_time(elapsed_time: float):
        print(f"Execution time: {elapsed_time} [ms]")

    def plot_returns(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.data_to_learn_and_validate)
        plt.ylabel('Return', fontsize=20)

    def plot_pacf(self):
        plot_pacf(self.data_to_learn)
        plt.show()

    def plot_acf(self):
        plot_acf(self.data_to_learn)
        plt.show()

    def plot_extrapolation(self, prediction):
        utils.plot_extrapolation(self, prediction, PredictionMethod.Arima)


class ManualArima(ArimaPrediction):
    def __init__(self, prices: Series, prediction_start: int, column: SeriesColumn, deviation: DeviationSource):
        super().__init__(prices, prediction_start, column, deviation)

    def extrapolate_and_measure(self, params: dict):
        return super().execute_and_measure(self.extrapolate, params)

    def find_d(self):
        return ndiffs(self.data_to_learn, test='adf')

    def extrapolate(self, params: dict):
        data_with_prediction = self.data_to_learn.copy()
        for date, r in self.data_to_learn_and_validate.iloc[self.prediction_start:].items():
            model = ARIMA(data_with_prediction,
                          order=(params.get("p", 1), self.find_d(), params.get("q", 1))).fit()

            single_prediction = model.forecast()
            prediction_series = pd.Series(single_prediction.values, index=[date])
            data_with_prediction = data_with_prediction.append(prediction_series)

        extrapolation = data_with_prediction[self.prediction_start:]
        return extrapolation


class AutoArima(ArimaPrediction):
    def __init__(self, prices: Series, prediction_start: int, column: SeriesColumn, deviation: DeviationSource):
        super().__init__(prices, prediction_start, column, deviation)
        self.auto_arima_model = None

    def extrapolate_and_measure(self, params: dict):
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict):
        self.auto_arima_model = auto_arima(self.data_to_learn,
                                           stat_p=params.get("p", 1),
                                           start_q=params.get("q", 1),
                                           test="adf",
                                           trace=True)
        periods = len(self.data_to_learn_and_validate) - self.prediction_start
        extrapolation = self.auto_arima_model.predict(n_periods=periods)
        return extrapolation

    def print_summary(self):
        print(self.auto_arima_model.summary())
