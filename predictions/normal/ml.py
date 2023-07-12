import time

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
from xgboost import XGBRegressor
from mlforecast import MLForecast

from predictions.prediction import Prediction, PredictionStats, PredictionResults
from predictions.utils import prepare_sf_dataframe, extract_predictions
from timeseries.enums import DeviationSource, DeviationScale


class XGBoostBase(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: int,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark, weights=weights)

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
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
    def evaluate_model(real_y, predicted_y) -> ndarray:
        return np.sum((real_y * 100 - predicted_y * 100) ** 2)

    def extrapolate(self, params: dict) -> PredictionResults:
        x = DataFrame({'X': np.linspace(0, self.training_size, self.training_size)})
        x_test = DataFrame({'X': np.linspace(self.training_size, self.train_and_pred_size, self.predict_size)})
        start_time = time.perf_counter_ns()

        if params.get("optimize", False):
            indices = DataFrame({'X': np.linspace(0, self.train_and_pred_size, self.train_and_pred_size)})
            x_opt, x_test_opt, y_opt, y_test_opt = train_test_split(indices, self.data_to_learn.values,
                                                                    test_size=self.predict_size,
                                                                    shuffle=False)
            space = self.optimization_space()

            @use_named_args(space)
            def objective(**params) -> ndarray:
                model = self.create_model(**params)
                model.fit(x_opt, y_opt)
                result_y = model.predict(x_test_opt)
                return self.evaluate_model(y_test_opt, result_y)

            optimization_result = gp_minimize(objective, space, n_calls=50, random_state=0)
            print(f'Best score: {optimization_result.fun}')
            best_params = {
                space[i].name: optimization_result.x[i] for i in range(len(optimization_result.x))
            }
            print(f'Best param values: {best_params}')

            model = self.create_model(**best_params)
        else:
            model = XGBRegressor(n_estimators=250)

        model.fit(x, self.data_to_learn)
        fit_time = time.perf_counter_ns()

        result = model.predict(x_test)
        prediction_time = time.perf_counter_ns()

        return PredictionResults(results=result,
                                 start_time=start_time, model_time=fit_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return XGBoostBase


class XGBoost(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: int,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark, weights=weights)

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> PredictionResults:
        df = prepare_sf_dataframe(self.data_to_learn, self.training_size)
        df["ds"] = pd.to_datetime(df["ds"])

        start_time = time.perf_counter_ns()
        models = [XGBRegressor(random_state=0, n_estimators=100)]
        model = MLForecast(models=models,
                           freq='D',
                           lags=[1, 7, 14],
                           lag_transforms={
                               1: [(rolling_mean, 7), (rolling_max, 7), (rolling_min, 7)],
                           },
                           date_features=['dayofweek', 'month'],
                           num_threads=6)
        model.fit(df, id_col='unique_id', time_col='ds', target_col='y', static_features=[])
        fit_time = time.perf_counter_ns()

        extrapolation = model.predict(horizon=self.predict_size)
        prediction_time = time.perf_counter_ns()

        results = extract_predictions(extrapolation, "XGBRegressor")
        return PredictionResults(results=results,
                                 start_time=start_time, model_time=fit_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return XGBoost
