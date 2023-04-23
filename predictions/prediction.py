import time

from pandas import Series

from predictions import utils
from timeseries.utils import DeviationSource, SeriesColumn


class PredictionResults:
    def __init__(self, elapsed_time: float, rmse: float, mae: float, mape: float):
        self.elapsed_time = elapsed_time
        self.rmse = rmse
        self.mae = mae
        self.mape = mape


class Prediction:
    def __init__(self, prices: Series, real_prices: Series, training_set_end: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource):
        self.data_to_learn = prices[:training_set_end].dropna()
        self.data_to_validate = Series(real_prices.values[training_set_end:])
        self.data_to_learn_and_validate = self.data_to_learn.append(self.data_to_validate)
        self.data_size = len(self.data_to_learn_and_validate)
        self.training_set_end = len(self.data_to_learn)
        self.prediction_delay = prediction_delay
        self.prediction_start = self.training_set_end + prediction_delay
        self.column = column
        self.deviation = deviation

    def execute_and_measure(self, extrapolation_method, params: dict) -> PredictionResults:
        start_time = time.time_ns()
        extrapolation = extrapolation_method(params)
        elapsed_time = round((time.time_ns() - start_time) / 1e6)

        rmse = utils.calculate_rmse(self, extrapolation)
        mae = utils.calculate_mae(self, extrapolation)
        mape = utils.calculate_mape(self, extrapolation)
        results = PredictionResults(elapsed_time, rmse, mae, mape)

        return results
