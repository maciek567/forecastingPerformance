import time

from pyspark.sql import DataFrame

from predictions import utils
from timeseries.utils import DeviationSource, SeriesColumn


class PredictionResultsHPC:
    def __init__(self, elapsed_time: float, rmse: float, mae: float, mape: float):
        self.elapsed_time = elapsed_time
        self.mitigation_time = 0.0
        self.rmse = rmse
        self.mae = mae
        self.mape = mape


class PredictionHPC:
    def __init__(self, prices: DataFrame, real_prices: DataFrame, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, mitigation_time: int = 0):
        self.data_to_learn = prices.limit(prediction_border).dropna()
        self.training_set_end = self.data_to_learn.count()
        self.prediction_delay = prediction_delay
        self.prediction_start = self.training_set_end + prediction_delay
        self.data_to_validate = real_prices.subtract(self.data_to_learn)
        self.data_to_learn_and_validate = self.data_to_learn.union(self.data_to_validate)
        self.data_size = self.data_to_learn_and_validate.count()
        self.column = column
        self.deviation = deviation
        self.mitigation_time = mitigation_time

    def execute_and_measure(self, extrapolation_method, params: dict) -> PredictionResultsHPC:
        start_time = time.time_ns()
        extrapolation = extrapolation_method(params)
        elapsed_time_ms = (time.time_ns() - start_time) / 1e6

        rmse = utils.calculate_rmse(self.data_to_validate.values, extrapolation)
        mae = utils.calculate_mae(self.data_to_validate.values, extrapolation)
        mape = utils.calculate_mape(self.data_to_validate.values, extrapolation)
        results = PredictionResultsHPC(elapsed_time_ms, rmse, mae, mape)

        return results
