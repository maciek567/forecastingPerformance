import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from metrics.utils import DefectsSource
from timeseries.utils import SeriesColumn


class ArimaPrediction:
    def __init__(self, prices: Series, prediction_start: int, column: SeriesColumn, defect: DefectsSource):
        self.prediction_start = prediction_start
        returns = prices.dropna()
        self.data_to_learn = returns[:prediction_start]
        self.data_to_learn_and_validate = returns
        self.column = column
        self.defect = defect

    def execute_and_measure(self, extrapolation_method, params: dict):
        start_time = time.time_ns()
        extrapolation = extrapolation_method(params)
        elapsed_time = round((time.time_ns() - start_time) / 1e6)
        rms = self.calculate_rms(extrapolation)
        return elapsed_time, rms

    @staticmethod
    def print_elapsed_time(elapsed_time: float):
        print(f"Execution time: {elapsed_time} [ms]")

    def calculate_rms(self, prediction: Series):
        actual_data = self.data_to_learn_and_validate[self.prediction_start:]
        return round(np.sqrt(np.mean((prediction - actual_data) ** 2)), 3)

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

    def plot_extrapolation(self, prediction: Series):
        plt.plot(self.data_to_learn_and_validate.values, "r", label="Actual data")
        plt.plot(range(self.prediction_start, len(self.data_to_learn_and_validate)), prediction, "b",
                 label="Prediction")
        plt.axvline(x=self.prediction_start, color='g', label='Extrapolation start')
        plt.title(f"Arima prediction [{self.column.value} prices, {self.defect.value}]")
        plt.xlabel("Time [days]")
        plt.ylabel("Prices [USD]")
        plt.legend()
        plt.show()


class ManualArima(ArimaPrediction):
    def __init__(self, prices: Series, prediction_start: int, column: SeriesColumn, defect: DefectsSource):
        super().__init__(prices, prediction_start, column, defect)

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
    def __init__(self, prices: Series, prediction_start: int, column: SeriesColumn, defect: DefectsSource):
        super().__init__(prices, prediction_start, column, defect)
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
