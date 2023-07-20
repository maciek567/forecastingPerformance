import gc
import time

from pandas import Series

from predictions import utils
from predictions.utils import normalized_columns_weights
from timeseries.enums import DeviationSource, DeviationScale


class PredictionResults:
    def __init__(self, results: dict, parameters: tuple = None,
                 start_time: float = 0.0, model_time: float = 0.0, prediction_time: float = 0.0):
        self.results = results
        self.parameters = parameters
        self.model_time = (model_time - start_time) / 1e6
        self.prediction_time = (prediction_time - model_time) / 1e6


class PredictionResSimple:
    def __init__(self, results: dict, parameters: tuple = None,
                 model_time: float = 0.0, prediction_time: float = 0.0):
        self.results = results
        self.parameters = parameters
        self.model_time = model_time
        self.prediction_time = prediction_time


class PredictionStats:
    def __init__(self, results: dict, parameters: tuple = None, start_time: float = 0.0,
                 elapsed_time: float = 0.0, model_time: float = 0.0, prediction_time: float = 0.0,
                 rmse: float = None, mae: float = None, mape: float = None):
        self.results = results
        self.parameters = parameters
        self.prepare_time = (elapsed_time - start_time) / 1e6 - model_time - prediction_time
        self.model_time = model_time
        self.prediction_time = prediction_time
        self.mitigation_time = 0.0
        self.rmse = rmse
        self.mae = mae
        self.mape = mape


class Prediction:
    def __init__(self, prices_dict: dict, real_prices_dict: dict, prediction_border: int, prediction_delay: dict,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time_dict: dict = None,
                 spark=None, weights: dict = None):
        self.data_with_defects = {column: prices[:prediction_border - prediction_delay[column]].values
                                  for column, prices in prices_dict.items()}
        self.data_to_learn = {column: prices[:prediction_border - prediction_delay[column]].dropna()
                              for column, prices in prices_dict.items()}
        self.training_size = {column: len(prices.values) for column, prices in self.data_to_learn.items()}
        self.prediction_start = prediction_border
        self.prediction_delay = prediction_delay
        self.training_end = {column: prediction_border - prediction_delay[column] for column in columns}
        self.data_to_validate = {column: Series(real_prices.values[prediction_border:]) for column, real_prices in
                                 real_prices_dict.items()}
        self.actual_data = real_prices_dict
        self.validation_size = len(list(self.data_to_validate.values())[0])
        self.predict_size = self.validation_size + max(list(prediction_delay.values()))
        self.columns = columns
        self.weights = weights
        self.deviation = deviation
        self.scale = scale
        self.mitigation_time = mitigation_time_dict
        self.spark = spark

    def execute_and_measure(self, extrapolation_method, params: dict) -> PredictionStats:
        gc.disable()
        start_time = time.perf_counter_ns()
        extrapolation = extrapolation_method(params)
        elapsed_time = time.perf_counter_ns()
        gc.enable()

        weights = normalized_columns_weights(self.columns, self.weights)
        rmse, mae, mape = 0.0, 0.0, 0.0
        for column, series in self.data_to_validate.items():
            rmse += utils.calculate_rmse(series.values, extrapolation.results[column].values) * weights[column]
            mae += utils.calculate_mae(series.values, extrapolation.results[column].values) * weights[column]
            mape += utils.calculate_mape(series.values, extrapolation.results[column].values) * weights[column]

        results = PredictionStats(results=extrapolation.results, parameters=extrapolation.parameters,
                                  start_time=start_time, elapsed_time=elapsed_time,
                                  model_time=extrapolation.model_time, prediction_time=extrapolation.prediction_time,
                                  rmse=rmse, mae=mae, mape=mape)

        return results
