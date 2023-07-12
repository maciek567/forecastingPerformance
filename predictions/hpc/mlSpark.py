import time

import numpy as np
import pandas as pd
from mlforecast.distributed import DistributedMLForecast
from mlforecast.distributed.models.spark.xgb import SparkXGBForecast
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import GBTRegressor
from pyspark.sql.functions import monotonically_increasing_id
from window_ops.rolling import rolling_mean, rolling_max, rolling_min

from predictions.prediction import PredictionResults
from predictions.prediction import PredictionStats, Prediction
from predictions.utils import prepare_sf_dataframe, extract_predictions, prepare_spark_dataframe
from timeseries.enums import DeviationSource, DeviationScale


class GBTRegressorSpark(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: int,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark, weights=weights)

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> PredictionResults:
        learn_id = self.data_to_learn.select("*").withColumn("id", monotonically_increasing_id())
        validate_id = self.data_to_validate.select("*").withColumn("id",
                                                                   self.train_and_pred_size + monotonically_increasing_id())
        train = learn_id.rdd.map(lambda x: (Vectors.dense(float(x[2])), x[1])).toDF(["features", "label"])
        test = validate_id.rdd.map(lambda x: (Vectors.dense(float(x[2])), x[1])).toDF(["features", "label"])

        start_time = time.perf_counter_ns()
        gbt = GBTRegressor(maxDepth=2, seed=42, leafCol="leafId")
        gbt.setMaxIter(5)
        gbt.setMinWeightFractionPerNode(0.049)
        model = gbt.fit(train)
        fit_time = time.perf_counter_ns()

        result = model.transform(test).head(self.validation_size)
        prediction_time = time.perf_counter_ns()

        result = np.array([row[2] for row in result])
        return PredictionResults(results=result,
                               start_time=start_time, model_time=fit_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return GBTRegressorSpark


class XGBoostSpark(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: int,
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

        extrapolation_df = extrapolation.toPandas()[-self.predict_size:]
        results = extract_predictions(extrapolation_df, "SparkXGBForecast")
        return PredictionResults(results=results,
                                 start_time=start_time, model_time=fit_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return XGBoostSpark
