from pyspark.sql import SparkSession

from predictions.hpc.arimaSpark import AutoArimaSpark
from predictions.hpc.mlSpark import GBTRegressorSpark
from predictions.hpc.statisticalSpark import CesSpark


def handle_spark(method):
    spark_methods = [AutoArimaSpark, CesSpark, GBTRegressorSpark]
    return start_spark() if method in spark_methods else None


def start_spark():
    return SparkSession.builder.appName("Predictions").getOrCreate()
