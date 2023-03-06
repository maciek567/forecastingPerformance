import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import time


class ArimaPrediction:

    def __init__(self, prices, prediction_start, prediction_end):
        self.prices = prices
        self.prediction_start = prediction_start
        self.prediction_end = prediction_end
        self.returns = prices.pct_change().dropna()
        self.data_to_learn = self.returns[:prediction_start]
        self.data_to_learn_and_validate = self.returns[:prediction_end]

    def plot_prices(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.prices)
        plt.ylabel('Prices', fontsize=20)

    def plot_returns(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.returns)
        plt.ylabel('Return', fontsize=20)

    def plot_pacf(self):
        plot_pacf(self.returns)
        plt.show()

    def plot_acf(self):
        plot_acf(self.returns)
        plt.show()

    def extrapolate(self, order):
        start_time = time.time_ns()
        data_with_prediction = self.data_to_learn.copy()
        for date, r in self.returns.iloc[self.prediction_start:self.prediction_end].items():
            model = ARIMA(data_with_prediction, order=order).fit()

            single_prediction = model.forecast()
            prediction_series = pd.Series(single_prediction.values, index=[date])
            data_with_prediction = data_with_prediction.append(prediction_series)

        elapsed_time = (time.time_ns() - start_time) / 1e6
        extrapolation = data_with_prediction[self.prediction_start:self.prediction_end]
        return extrapolation, elapsed_time

    def plot_extrapolation(self, prediction):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.data_to_learn_and_validate, label="Actual data")
        ax.plot(prediction, label="prediction")
        # plt.axvline(x = prediction_start, color = 'g', label = 'extrapolation start')
        ax.set_xlabel("Date")
        ax.set_ylabel("Percentage price change")
        plt.show()

    @staticmethod
    def print_elapsed_time(elapsed_time):
        print(f"Execution time: {round(elapsed_time, 2)} [ms]")

    def print_rms(self, prediction):
        actual_data = self.returns[self.prediction_start:self.prediction_end]
        print("RMS: %r " % round(np.sqrt(np.mean((prediction - actual_data) ** 2)), 3))
