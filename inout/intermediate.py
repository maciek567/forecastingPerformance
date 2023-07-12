import glob
import os

from pandas import Series
from pyspark.sql import DataFrame

from inout.paths import deviations_path
from timeseries.enums import DeviationScale, DeviationSource, SeriesColumn
from timeseries.utils import Mitigation


class IntermediateProvider:

    def __init__(self, spark=None):
        self.spark = spark

    @staticmethod
    def save_as_csv(series: Series, name: str):
        os.makedirs(deviations_path, exist_ok=True)
        series.to_csv(IntermediateProvider.get_csv_path(name), index_label=["Date"], header=["Values"])

    @staticmethod
    def save_as_latex(path, latex):
        latex_file = open(path + ".tex", "w")
        latex_file.write(latex)
        latex_file.close()

    def load_csv_as_spark(self, name: str) -> DataFrame:
        df = self.spark.read.option("header", "true").csv(IntermediateProvider.get_csv_path(name))
        df = df.withColumn("Values", df.Values.cast("float"))
        return df

    @staticmethod
    def get_csv_path(name: str) -> str:
        return os.path.join(deviations_path, name) + ".csv"

    @staticmethod
    def remove_current_files():
        files = glob.glob(deviations_path + "*")
        for f in files:
            os.remove(f)

    def load_set_as_spark(self, is_mitigated: bool, sources: list, scales: list) -> dict:
        series_set = {deviation: {scale: {attribute: None for attribute in SeriesColumn} for scale in scales} for
                      deviation in sources}
        for file in os.listdir(deviations_path):
            name = file.strip(".csv")
            parts = name.split("_")
            series = self.spark.read.option("header", "true").csv(deviations_path + file)
            series = series.withColumn("Values", series.Values.cast("float"))
            if not is_mitigated and parts[-1] == "deviated":
                series_set[DeviationSource[parts[1]]][DeviationScale[parts[2]]][SeriesColumn[parts[3]]] = series
            elif is_mitigated and parts[-1] == "mitigated":
                mitigation = series_set[DeviationSource[parts[1]]][DeviationScale[parts[2]]][
                    SeriesColumn[parts[3]]] = {}
                mitigation[Mitigation.DATA] = series
                mitigation[Mitigation.TIME] = float(parts[4])

        return series_set
