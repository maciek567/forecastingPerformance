import numpy as np
from numpy import ndarray
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame
from pyspark.sql.functions import monotonically_increasing_id

from predictions.hpc.predictionHPC import PredictionHPC, PredictionResultsHPC
from timeseries.enums import SeriesColumn, DeviationSource


class GBTRegressorHPC(PredictionHPC):
    def __init__(self, prices: DataFrame, real_prices: DataFrame, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, mitigation_time: int = 0, spark=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, mitigation_time,
                         spark)

    def extrapolate_and_measure(self, params: dict) -> PredictionResultsHPC:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> ndarray:
        learn_id = self.data_to_learn.select("*").withColumn("id", monotonically_increasing_id())
        validate_id = self.data_to_validate.select("*").withColumn("id", self.data_size + monotonically_increasing_id())
        train = learn_id.rdd.map(lambda x: (Vectors.dense(float(x[2])), x[1])).toDF(["features", "label"])
        test = validate_id.rdd.map(lambda x: (Vectors.dense(float(x[2])), x[1])).toDF(["features", "label"])

        gbt = GBTRegressor(maxDepth=2, seed=42, leafCol="leafId")
        gbt.setMaxIter(5)
        gbt.setMinWeightFractionPerNode(0.049)
        model = gbt.fit(train)
        result = model.transform(test).head(self.validation_size)

        results = [row[2] for row in result]
        return np.array(results)
