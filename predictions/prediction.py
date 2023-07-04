import gc
import time

from numpy import ndarray
from pandas import Series, concat

from predictions import utils
from timeseries.enums import DeviationSource, SeriesColumn, DeviationScale


class PredictionResults:
    def __init__(self, results: ndarray, parameters: tuple = None,
                 start_time: float = 0.0, model_time: float = 0.0, prediction_time: float = 0.0):
        self.results = results
        self.parameters = parameters
        self.model_time = (model_time - start_time) / 1e6
        self.prediction_time = (prediction_time - model_time) / 1e6


class PredictionStats:
    def __init__(self, parameters: tuple = None, start_time: float = 0.0,
                 elapsed_time: float = 0.0, model_time: float = 0.0, prediction_time: float = 0.0,
                 rmse: float = None, mae: float = None, mape: float = None):
        self.parameters = parameters
        self.prepare_time = (elapsed_time - start_time) / 1e6 - model_time - prediction_time
        self.model_time = model_time
        self.prediction_time = prediction_time
        self.mitigation_time = 0.0
        self.rmse = rmse
        self.mae = mae
        self.mape = mape


class Prediction:
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, scale: DeviationScale, mitigation_time: int = 0,
                 spark=None):
        self.data_with_defects = prices[:prediction_border].values
        self.data_to_learn = prices[:prediction_border].dropna()
        self.training_size = len(self.data_to_learn)
        self.prediction_border = prediction_border
        self.prediction_delay = prediction_delay
        self.prediction_start = prediction_border + prediction_delay
        self.data_to_validate = Series(real_prices.values[self.prediction_start:])
        self.actual_data = real_prices
        self.predict_size = len(self.data_to_validate)
        self.train_and_pred_size = self.training_size + self.predict_size
        self.column = column
        self.deviation = deviation
        self.scale = scale
        self.mitigation_time = mitigation_time
        self.spark = spark

    def execute_and_measure(self, extrapolation_method, params: dict) -> PredictionStats:
        gc.disable()
        start_time = time.perf_counter_ns()
        prediction = extrapolation_method(params)
        elapsed_time = time.perf_counter_ns()
        gc.enable()

        rmse = utils.calculate_rmse(self.data_to_validate.values, prediction.results)
        mae = utils.calculate_mae(self.data_to_validate.values, prediction.results)
        mape = utils.calculate_mape(self.data_to_validate.values, prediction.results)
        results = PredictionStats(parameters=prediction.parameters,
                                  start_time=start_time, elapsed_time=elapsed_time,
                                  model_time=prediction.model_time, prediction_time=prediction.prediction_time,
                                  rmse=rmse, mae=mae, mape=mape)

        return results
