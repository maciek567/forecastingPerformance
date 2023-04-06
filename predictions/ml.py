import time

import auto_esn.utils.dataset_loader as dl
import numpy as np
import pandas as pd
import torch
from auto_esn.esn.esn import DeepESN
from numpy import ndarray
from pandas import Series
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from xgboost import XGBRegressor

from metrics.utils import DefectsSource
from predictions import utils
from timeseries.utils import SeriesColumn
from utils import PredictionMethod


class MlPrediction:
    def __init__(self, prices: Series, prediction_start: int, column: SeriesColumn, defect: DefectsSource):
        self.data_to_learn = prices.dropna()[:prediction_start]
        self.data_to_learn_and_validate = prices.dropna()
        self.data_size = len(self.data_to_learn_and_validate)
        self.prediction_start = prediction_start
        self.column = column
        self.defect = defect

    def execute_and_measure(self, extrapolation_method, params: dict):
        start_time = time.time_ns()
        extrapolation = extrapolation_method(params)
        elapsed_time = round((time.time_ns() - start_time) / 1e6)
        rms = utils.calculate_rms(self, extrapolation)
        return elapsed_time, rms


class Reservoir(MlPrediction):
    def __init__(self, prices: Series, prediction_start: int, column: SeriesColumn, defect: DefectsSource):
        super().__init__(prices, prediction_start, column, defect)

    def extrapolate_and_measure(self, params: dict):
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict):
        mg17 = dl.loader_explicit(utils.normalize(self.data_to_learn_and_validate),
                                  test_size=self.data_size - self.prediction_start)
        x, x_test, y, y_test = mg17()

        esn = DeepESN(num_layers=2,
                      hidden_size=100)
        esn.fit(x, y)

        predictions = []
        for j in range(self.data_size - self.prediction_start):
            point_to_predict = x_test[j: j + 1]
            predicted_value = esn(point_to_predict)
            predictions.append(predicted_value)

        res = torch.vstack(predictions)
        return utils.denormalize(res.numpy(), self.data_to_learn_and_validate)

    def plot_extrapolation(self, result) -> None:
        utils.plot_extrapolation(self, result, PredictionMethod.Reservoir)


class XGBoost(MlPrediction):
    def __init__(self, prices, prediction_start, column: SeriesColumn, defect: DefectsSource):
        super().__init__(prices, prediction_start, column, defect)

    def extrapolate_and_measure(self, params: dict):
        return super().execute_and_measure(self.extrapolate, params)

    @staticmethod
    def optimization_space() -> list:
        return [
            Integer(1, 5, name='max_depth'),
            Integer(15, 100, name='n_estimators'),
            Real(10 ** -5, 10 ** 0, 'log-uniform', name='learning_rate'),
            Real(10 ** -5, 10 ** 1, 'log-uniform', name='reg_alpha'),
            Real(10 ** -5, 10 ** 1, 'log-uniform', name='reg_lambda'),
        ]

    @staticmethod
    def create_model(**params):
        return XGBRegressor(
            objective='reg:squarederror',
            **params
        )

    @staticmethod
    def forecast(train_x, train_y, test_x, model) -> ndarray:
        model.fit(train_x, train_y)
        result_y = model.predict(test_x)
        return result_y

    @staticmethod
    def evaluate_model(real_y, predicted_y):
        return np.sum((real_y * 100 - predicted_y * 100) ** 2)

    def extrapolate(self, params: dict) -> list:
        indices = pd.DataFrame({'X': np.linspace(0, self.data_size, self.data_size)})
        x, x_test, y, y_test = train_test_split(indices, self.data_to_learn_and_validate.values,
                                                test_size=self.data_size - self.prediction_start,
                                                shuffle=False)

        if params.get("optimize", False):
            space = self.optimization_space()

            @use_named_args(space)
            def objective(**params) -> float:
                model = self.create_model(**params)
                result_y = self.forecast(x, y, x_test, model)
                return self.evaluate_model(y_test, result_y)

            optimization_result = gp_minimize(objective, space, n_calls=50, random_state=0)
            print(f'Best score: {optimization_result.fun}')
            best_params = {
                space[i].name: optimization_result.x[i] for i in range(len(optimization_result.x))
            }
            print(f'Best param values: {best_params}')

            model = self.create_model(**best_params)
        else:
            model = XGBRegressor(n_estimators=250)

        res = self.forecast(x, y, x_test, model)
        return res

    def plot_extrapolation(self, result) -> None:
        utils.plot_extrapolation(self, result, PredictionMethod.XGBoost)