import time

import numpy as np
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import GBTRegressor
from pyspark.sql.functions import monotonically_increasing_id

from predictions.prediction import PredictionStats, Prediction
from timeseries.enums import DeviationSource, DeviationScale


class GBTRegressorSpark(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: int,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark, weights=weights)

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> PredictionStats:
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
        return PredictionStats(results=result,
                               start_time=start_time, model_time=fit_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return GBTRegressorSpark
