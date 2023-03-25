import time

import auto_esn.utils.dataset_loader as dl
import numpy as np
import pandas as pd
import torch
from auto_esn.esn.esn import DeepESN
from matplotlib import pyplot as plt


class ReservoirResult:
    def __init__(self, X, y, X_test, y_test, res):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.res = res


class Reservoir:

    def __init__(self, prices, prediction_start, prediction_end):
        self.prices = prices
        self.min_val = pd.Series(prices).min()
        self.max_val = pd.Series(prices).max()
        self.normalized_prices = self.normalize()
        self.prediction_start = prediction_start
        self.prediction_end = prediction_end

    def normalize(self):
        df = pd.DataFrame((self.prices - self.min_val) / (self.max_val - self.min_val))
        df.columns = ["y"]
        return df

    def denormalize(self, prices):
        return prices * (self.max_val - self.min_val) + self.min_val

    def extrapolate_and_measure(self, params: dict):
        start_time = time.time_ns()
        extrapolation = self.extrapolate(params)
        elapsed_time = round((time.time_ns() - start_time) / 1e6)
        rms = self.calculate_rms(extrapolation)
        return elapsed_time, rms

    def calculate_rms(self, result: ReservoirResult):
        actual_data = self.prices[self.prediction_start:self.prediction_end].values
        prediction = (self.denormalize(result.res).numpy())
        return round(np.sqrt(np.mean((prediction - actual_data) ** 2)), 3)

    def extrapolate(self, params: dict) -> ReservoirResult:
        mg17 = dl.loader_explicit(self.normalized_prices, test_size=self.prediction_end - self.prediction_start)
        X, X_test, y, y_test = mg17()

        esn = DeepESN(num_layers=2,
                      hidden_size=100)

        esn.fit(X, y)

        point_to_predict = X_test[0:1]
        predictions = []
        for j in range(self.prediction_end - self.prediction_start):
            predicted_value = esn(point_to_predict)
            predictions.append(predicted_value)

        res = torch.vstack(predictions)
        return ReservoirResult(X, y, X_test, y_test, res)

    @staticmethod
    def plot_extrapolation(result: ReservoirResult):
        predicted = np.hstack(
            [result.X.view(-1).detach().numpy()[:result.X.shape[0]], result.res.view(-1).detach().numpy()])
        true = np.hstack(
            [result.y.view(-1).detach().numpy()[:result.X.shape[0]], result.y_test.view(-1).detach().numpy()])

        plt.plot(range(result.X.shape[0] + result.X_test.shape[0]), predicted, 'r', label='predicted')
        plt.plot(range(result.X.shape[0] + result.X_test.shape[0]), true, 'b', label='true')
        plt.axvline(x=result.X.shape[0], color='g', label='extrapolation start')
        plt.legend()
        plt.show()
