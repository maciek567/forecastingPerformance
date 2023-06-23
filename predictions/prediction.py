import time

from pandas import Series, concat

from predictions import utils
from timeseries.enums import DeviationSource, SeriesColumn


class PredictionResults:
    def __init__(self, elapsed_time: float, rmse: float, mae: float, mape: float):
        self.elapsed_time = elapsed_time
        self.mitigation_time = 0.0
        self.rmse = rmse
        self.mae = mae
        self.mape = mape


class Prediction:
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, mitigation_time: int = 0, spark=None):
        self.data_to_learn = prices[:prediction_border].dropna()
        self.training_set_end = len(self.data_to_learn)
        self.prediction_delay = prediction_delay
        self.prediction_start = self.training_set_end + prediction_delay
        self.data_to_validate = Series(real_prices.values[self.prediction_start:])
        self.data_to_learn_and_validate = concat([self.data_to_learn, self.data_to_validate])
        self.data_size = len(self.data_to_learn_and_validate)
        self.column = column
        self.deviation = deviation
        self.mitigation_time = mitigation_time
        self.spark = spark

    def execute_and_measure(self, extrapolation_method, params: dict) -> PredictionResults:
        start_time = time.time_ns()
        extrapolation = extrapolation_method(params)
        elapsed_time_ms = (time.time_ns() - start_time) / 1e6

        rmse = utils.calculate_rmse(self.data_to_validate.values, extrapolation)
        mae = utils.calculate_mae(self.data_to_validate.values, extrapolation)
        mape = utils.calculate_mape(self.data_to_validate.values, extrapolation)
        results = PredictionResults(elapsed_time_ms, rmse, mae, mape)

        return results
