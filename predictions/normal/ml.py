import auto_esn.utils.dataset_loader as dl
import numpy as np
import torch
from auto_esn.esn.esn import DeepESN
from numpy import ndarray
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from xgboost import XGBRegressor

from predictions import utils
from predictions.prediction import Prediction, PredictionResults
from timeseries.enums import SeriesColumn, DeviationSource


class Reservoir(Prediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, mitigation_time: int = 0, spark=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, mitigation_time,
                         spark)

    def extrapolate_and_measure(self, params: dict) -> PredictionResults:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> PredictionResults:
        train = dl.loader_explicit(utils.normalize(self.data_to_learn, self.data_to_learn), test_size=0)
        _, x_train, _, y_train = train()
        test = dl.loader_explicit(utils.normalize(self.data_to_validate, self.data_to_learn), test_size=0)
        _, x_test, _, y_test = test()

        esn = DeepESN(num_layers=2,
                      hidden_size=100)
        esn.fit(x_train, y_train)

        predictions = []
        for j in range(self.data_size - self.training_set_end):
            point_to_predict = x_test[j: j + 1]
            predicted_value = esn(point_to_predict)
            predictions.append(predicted_value)

        result = [x[0] for x in torch.vstack(predictions).tolist()]
        if len(self.data_to_learn) + len(result) < self.data_size:
            self.data_size = len(self.data_to_learn) + len(result)
            self.data_to_validate = self.data_to_validate[0:-2]

        result = utils.denormalize(np.array(result), self.data_to_learn)
        return PredictionResults(results=result)

    def plot_extrapolation(self, prediction, company_name, save_file: bool = False):
        utils.plot_extrapolation(self, prediction, Reservoir, company_name, save_file)


class XGBoost(Prediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, mitigation_time: int = 0, spark=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, mitigation_time,
                         spark)

    def extrapolate_and_measure(self, params: dict) -> PredictionResults:
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
    def evaluate_model(real_y, predicted_y) -> ndarray:
        return np.sum((real_y * 100 - predicted_y * 100) ** 2)

    def extrapolate(self, params: dict) -> PredictionResults:
        indices = DataFrame({'X': np.linspace(0, self.data_size, self.data_size)})
        x, x_test, y, y_test = train_test_split(indices, self.data_to_learn_and_validate.values,
                                                test_size=self.data_size - self.training_set_end,
                                                shuffle=False)

        if params.get("optimize", False):
            space = self.optimization_space()

            @use_named_args(space)
            def objective(**params) -> ndarray:
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

        result = self.forecast(x, y, x_test, model)
        return PredictionResults(results=result)

    def plot_extrapolation(self, prediction, company_name, save_file: bool = False):
        utils.plot_extrapolation(self, prediction, XGBoost, company_name, save_file)
