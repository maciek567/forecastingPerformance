import time

import auto_esn.utils.dataset_loader as dl
import numpy as np
import pandas as pd
import torch
from auto_esn.esn.esn import DeepESN
from matplotlib import pyplot as plt

from metrics.utils import DefectsSource
from timeseries.utils import SeriesColumn


class ReservoirResult:
    def __init__(self, X, y, X_test, y_test, res):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.res = res


class Reservoir:

    def __init__(self, prices, prediction_start, column: SeriesColumn, defect: DefectsSource):
        self.prediction_start = prediction_start
        returns = prices.dropna()
        self.data_to_learn = self.normalize(returns[:prediction_start])
        self.data_to_learn_and_validate = self.normalize(returns)
        self.column = column
        self.defect = defect

    @staticmethod
    def normalize(series):
        df = pd.DataFrame((series - series.min()) / (series.max() - series.min()))
        df.columns = ["y"]
        return df

    @staticmethod
    def denormalize(series):
        return series * (series.max() - series.min()) + series.min()

    def extrapolate_and_measure(self, params: dict):
        start_time = time.time_ns()
        extrapolation = self.extrapolate(params)
        elapsed_time = round((time.time_ns() - start_time) / 1e6)
        rms = self.calculate_rms(extrapolation)
        return elapsed_time, rms

    def calculate_rms(self, result: ReservoirResult):
        actual_data = self.data_to_learn_and_validate[self.prediction_start:].values
        prediction = (self.denormalize(result.res).numpy())
        return round(np.sqrt(np.mean((prediction - actual_data) ** 2)), 3)

    def extrapolate(self, params: dict) -> ReservoirResult:
        mg17 = dl.loader_explicit(self.data_to_learn_and_validate,
                                  test_size=len(self.data_to_learn_and_validate) - self.prediction_start)
        X, X_test, y, y_test = mg17()

        esn = DeepESN(num_layers=2,
                      hidden_size=100)
        esn.fit(X, y)

        predictions = []
        for j in range(len(self.data_to_learn_and_validate) - self.prediction_start):
            point_to_predict = X_test[j: j + 1]
            predicted_value = esn(point_to_predict)
            predictions.append(predicted_value)

        res = torch.vstack(predictions)
        return ReservoirResult(X, y, X_test, y_test, res)

    def plot_extrapolation(self, result: ReservoirResult):
        true = np.hstack(
            [result.y.view(-1).detach().numpy()[:result.X.shape[0]], result.y_test.view(-1).detach().numpy()])
        predicted = np.hstack(
            [result.res.view(-1).detach().numpy()])

        plt.plot(range(result.X.shape[0] + result.X_test.shape[0]), true, 'r', label='Actual data')
        plt.plot(range(result.X.shape[0], result.X.shape[0] + result.X_test.shape[0]), predicted, 'b', label='Prediction')
        plt.axvline(x=result.X.shape[0], color='g', label='Extrapolation start')
        plt.title(f"Reservoir prediction [{self.column.value} prices, {self.defect.value}]")
        plt.xlabel("Time [days]")
        plt.ylabel("Normalized prices [USD]")
        plt.legend()
        plt.show()
