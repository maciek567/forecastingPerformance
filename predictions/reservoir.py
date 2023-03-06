import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import auto_esn.utils.dataset_loader as dl
from auto_esn.datasets.df import MackeyGlass
from auto_esn.esn.esn import GroupedDeepESN, DeepESN
from auto_esn.esn.reservoir.util import NRMSELoss
from auto_esn.esn.reservoir.activation import tanh
import time


class Reservoir:

    def __init__(self, prices, prediction_start, prediction_end):
        self.prices = prices
        self.prediction_start = prediction_start
        self.prediction_end = prediction_end

    def normalize(self):
        min_val = pd.Series(self.prices).min()
        max_val = pd.Series(self.prices).max()
        df = pd.DataFrame((self.prices - min_val) / (max_val - min_val))
        df.columns = ["y"]
        self.prices = df

    def get_rms(self, true, predicted):
        return round(np.sqrt(np.mean((predicted - true) ** 2)), 3)

    def extrapolate(self):
        start_time = time.time_ns()
        mg17 = dl.loader_explicit(self.prices, test_size=self.prediction_end-self.prediction_start)
        X, X_test, y, y_test = mg17()

        esn = DeepESN(num_layers=2,
                      hidden_size=100)

        # fit
        esn.fit(X, y)

        # esn already has the state after consuming whole training dataset
        # let's start from first element in test dataset and let it extrapolate further
        val = X_test[0:1]
        result = []
        for j in range(self.prediction_end-self.prediction_start):
            val = esn(val)
            result.append(val)

        res = torch.vstack(result)
        elapsed_time = (time.time_ns() - start_time) / 1e6
        print(f"Execution time: {round(elapsed_time, 2)} [ms]")

        return X, y, res, X_test, y_test, elapsed_time

    def plot(self, X, y, res, X_test, y_test):
        # plot validation set
        predicted = np.hstack([X.view(-1).detach().numpy()[:X.shape[0]], res.view(-1).detach().numpy()])
        true = np.hstack([y.view(-1).detach().numpy()[:X.shape[0]], y_test.view(-1).detach().numpy()])

        plt.plot(range(X.shape[0] + X_test.shape[0]), predicted, 'r', label='predicted')
        plt.plot(range(X.shape[0] + X_test.shape[0]), true, 'b', label='true')
        plt.axvline(x=X.shape[0], color='g', label='extrapolation start')
        plt.legend()
        plt.show()
