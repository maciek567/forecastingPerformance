import time

from pandas import Series
from statsforecast import StatsForecast
from statsforecast.models import AutoCES

from predictions.prediction import Prediction, PredictionStats, PredictionResults
from predictions.utils import prepare_sf_dataframe
from timeseries.enums import SeriesColumn, DeviationSource, DeviationScale


class Ces(Prediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, scale: DeviationScale, mitigation_time: int = 0,
                 spark=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, scale,
                         mitigation_time, spark)

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> PredictionResults:
        series = prepare_sf_dataframe(self.data_to_learn, self.training_size)

        start_time = time.perf_counter_ns()
        sf = StatsForecast(
            models=[AutoCES()],
            freq='D',
        )
        sf.fit(df=series)
        fit_time = time.perf_counter_ns()

        extrapolation = sf.predict(h=self.predict_size)
        prediction_time = time.perf_counter_ns()

        result = extrapolation.values[:, 1]
        return PredictionResults(results=result,
                                 start_time=start_time, model_time=fit_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return Ces
