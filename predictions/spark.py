from pyspark.sql import SparkSession

from predictions.hpc.mlSpark import GBTRegressorSpark, XGBoostSpark
from predictions.hpc.statsSpark import AutoArimaSpark
from predictions.hpc.statsSpark import CesSpark


def handle_spark(method):
    spark_methods = [AutoArimaSpark, CesSpark, GBTRegressorSpark, XGBoostSpark]
    return start_spark() if method in spark_methods else None


def start_spark():
    return SparkSession.builder.appName("Predictions").getOrCreate()
