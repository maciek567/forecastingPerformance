from numpy import ndarray
from pandas import Series, DataFrame
from statsforecast import StatsForecast
from statsforecast.models import AutoCES

from predictions import utils
from predictions.prediction import Prediction, PredictionResults
from timeseries.enums import SeriesColumn, DeviationSource


class Ces(Prediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, mitigation_time: int = 0, spark=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, mitigation_time,
                         spark)

    def extrapolate_and_measure(self, params: dict) -> PredictionResults:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> ndarray:
        series_id = [0 for i in range(0, self.training_set_end)]
        series = DataFrame({"ds": self.data_to_learn.keys(), "y": self.data_to_learn.values, "unique_id": series_id})

        sf = StatsForecast(
            models=[AutoCES()],
            freq='D',
        )
        extrapolation = sf.forecast(df=series, h=self.data_size - self.training_set_end)

        return extrapolation.values[:, 1]

    def plot_extrapolation(self, prediction, save_file: bool = False):
        utils.plot_extrapolation(self, prediction, Ces, save_file)
