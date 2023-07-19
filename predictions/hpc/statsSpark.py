import time

from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.models import AutoCES

from predictions.prediction import Prediction, PredictionStats, PredictionResults
from predictions.utils import prepare_sf_dataframe, prepare_spark_dataframe, extract_predictions
from timeseries.enums import DeviationSource, DeviationScale


class AutoArimaSpark(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: dict,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark, weights=weights)

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    @staticmethod
    def create_model():
        return StatsForecast(
            models=[AutoARIMA()],
            freq='D',
        )

    def extrapolate(self, params: dict) -> PredictionResults:
        df = prepare_sf_dataframe(self.data_to_learn, self.training_size)
        sdf = prepare_spark_dataframe(df, self.spark)

        start_time = time.perf_counter_ns()
        sf_fit = self.create_model()
        sf_fit.fit(df=df)
        params = sf_fit.fitted_[0][0].model_['arma']
        fit_time = time.perf_counter_ns()

        sf = self.create_model()
        extrapolation = sf.forecast(df=sdf, h=self.predict_size)
        prediction_time = time.perf_counter_ns()

        result = extract_predictions(extrapolation.toPandas(), "AutoARIMA")
        return PredictionResults(results=result, parameters=params,
                                 start_time=start_time, model_time=fit_time,
                                 prediction_time=prediction_time - (fit_time - start_time))

    @staticmethod
    def get_method():
        return AutoArimaSpark


class CesSpark(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: dict,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark, weights=weights)

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

        result = extract_predictions(extrapolation.toPandas(), "CES")
        return PredictionResults(results=result,
                                 start_time=start_time, model_time=fit_time,
                                 prediction_time=prediction_time - (fit_time - start_time))

    @staticmethod
    def get_method():
        return CesSpark
