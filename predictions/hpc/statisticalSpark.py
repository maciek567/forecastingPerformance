import time

from pandas import Series
from statsforecast import StatsForecast
from statsforecast.models import AutoCES

from predictions.prediction import Prediction, PredictionStats, PredictionResults
from predictions.utils import prepare_sf_dataframe, prepare_spark_dataframe
from timeseries.enums import SeriesColumn, DeviationSource, DeviationScale


class CesSpark(Prediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, scale: DeviationScale, mitigation_time: int = 0,
                 spark=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, scale,
                         mitigation_time, spark)

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    @staticmethod
    def create_model():
        return StatsForecast(
            models=[AutoCES()],
            freq='D',
        )

    def extrapolate(self, params: dict) -> PredictionResults:
        df = prepare_sf_dataframe(self.data_to_learn, self.training_size)
        sdf = prepare_spark_dataframe(df, self.spark)

        start_time = time.perf_counter_ns()
        sf_fit = self.create_model()
        sf_fit.fit(df=sdf)
        fit_time = time.perf_counter_ns()

        sf = self.create_model()
        extrapolation = sf.forecast(df=sdf, h=self.predict_size)
        prediction_time = time.perf_counter_ns()

        result = extrapolation.toPandas()["CES"]
        return PredictionResults(results=result,
                                 start_time=start_time, model_time=fit_time,
                                 prediction_time=prediction_time - (fit_time - start_time))

    @staticmethod
    def get_method():
        return CesSpark
