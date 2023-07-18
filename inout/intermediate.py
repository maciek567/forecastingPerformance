import glob
import os

from pandas import Series, DataFrame

from inout.paths import deviations_csv_path
from timeseries.enums import DeviationScale, DeviationSource, SeriesColumn
from timeseries.utils import MitigationType


class IntermediateProvider:

    def __init__(self, spark=None):
        self.spark = spark

    @staticmethod
    def save_as_csv(series: Series, path: str, name: str):
        os.makedirs(path, exist_ok=True)
        series.to_csv(IntermediateProvider.get_csv_path(name), index_label=["Date"], header=["Values"])

    @staticmethod
    def save_latex(latex: str, path: str, name: str):
        os.makedirs(path, exist_ok=True)
        latex_file = open(os.path.join(path, name) + ".tex", "w")
        latex_file.write(latex)
        latex_file.close()

    @staticmethod
    def save_as_tex(df: DataFrame, path: str, name: str):
        os.makedirs(path, exist_ok=True)
        latex = df.to_latex(index=False,
                            formatters={"name": str.upper},
                            float_format="{:.2f}".format)

        latex_file = open(os.path.join(path, name), "w")
        latex_file.write(latex)
        latex_file.close()

    def load_csv_as_spark(self, name: str) -> DataFrame:
        df = self.spark.read.option("header", "true").csv(IntermediateProvider.get_csv_path(name))
        df = df.withColumn("Values", df.Values.cast("float"))
        return df

    @staticmethod
    def get_csv_path(name: str) -> str:
        return os.path.join(deviations_csv_path, name) + ".csv"

    @staticmethod
    def remove_current_files():
        files = glob.glob(deviations_csv_path + "*")
        for f in files:
            os.remove(f)

    def load_set_as_spark(self, is_mitigated: bool, sources: list, scales: list) -> dict:
        series_set = {deviation: {scale: {attribute: None for attribute in SeriesColumn} for scale in scales} for
                      deviation in sources}
        for file in os.listdir(deviations_csv_path):
            name = file.strip(".csv")
            parts = name.split("_")
            series = self.spark.read.option("header", "true").csv(deviations_csv_path + file)
            series = series.withColumn("Values", series.Values.cast("float"))
            if not is_mitigated and parts[-1] == "deviated":
                series_set[DeviationSource[parts[1]]][DeviationScale[parts[2]]][SeriesColumn[parts[3]]] = series
            elif is_mitigated and parts[-1] == "mitigated":
                mitigation = series_set[DeviationSource[parts[1]]][DeviationScale[parts[2]]][
                    SeriesColumn[parts[3]]] = {}
                mitigation[MitigationType.DATA] = series
                mitigation[MitigationType.TIME] = float(parts[4])

        return series_set
