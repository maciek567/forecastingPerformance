import time

import pandas as pd
from mlforecast.distributed import DistributedMLForecast
from mlforecast.distributed.models.spark.xgb import SparkXGBForecast
from window_ops.rolling import rolling_mean, rolling_max, rolling_min

from predictions.prediction import PredictionResults
from predictions.prediction import PredictionStats, Prediction
from predictions.utils import prepare_sf_dataframe, extract_predictions, prepare_spark_dataframe, cut_extrapolation
from timeseries.enums import DeviationSource, DeviationScale


class XGBoostSpark(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: dict,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark, weights=weights)

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> PredictionResults:
        df = prepare_sf_dataframe(self.data_to_learn, self.training_size)
        df["ds"] = pd.to_datetime(df["ds"])
        sdf = prepare_spark_dataframe(df, self.spark)

        start_time = time.perf_counter_ns()
        models = [SparkXGBForecast(random_state=0)]
        fcst = DistributedMLForecast(models=models,
                                     freq='D',
                                     lags=[1, 7, 14],
                                     lag_transforms={
                                         1: [(rolling_mean, 7), (rolling_max, 7), (rolling_min, 7)],
                                     },
                                     date_features=['dayofweek', 'month'],
                                     num_threads=6,
                                     engine=self.spark)
        fcst.fit(sdf)
        fit_time = time.perf_counter_ns()

        extrapolation = fcst.predict(horizon=self.predict_size)
        prediction_time = time.perf_counter_ns()

        results_with_training = extract_predictions(extrapolation.toPandas(), "SparkXGBForecast")
        results = {column: results_with_training[column][-self.predict_size:] for column in self.columns}
        results = cut_extrapolation(results, self.prediction_delay, self.columns, self.data_to_validate)
        return PredictionResults(results=results,
                                 start_time=start_time, model_time=fit_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return XGBoostSpark
