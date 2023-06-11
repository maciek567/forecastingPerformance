import glob
import os

from pandas import Series, DataFrame, read_csv

from timeseries.utils import DeviationScale, DeviationSource, SeriesColumn, Mitigation

processed_path = "../data/processed/"


class IntermediateProvider:

    @staticmethod
    def save_as_csv(series: Series, name: str):
        series.to_csv(IntermediateProvider.get_csv_path(name))

    @staticmethod
    def load_csv(name: str) -> DataFrame:
        series = read_csv(IntermediateProvider.get_csv_path(name))
        series.index = series["Date"]
        series = series.drop("Date", axis=1)
        return series

    @staticmethod
    def get_csv_path(name: str) -> str:
        return processed_path + name + ".csv"

    @staticmethod
    def remove_current_files():
        files = glob.glob(processed_path + "*")
        for f in files:
            os.remove(f)

    @staticmethod
    def load_all(is_mitigated: bool, sources: list, scales: list):
        series_set = {deviation: {scale: {attribute: None for attribute in SeriesColumn} for scale in scales} for deviation in sources}
        for file in os.listdir(processed_path):
            name = file.strip(".csv")
            parts = name.split("_")
            series = read_csv(processed_path + file)
            if 'Date' in series.columns:
                series.index = series["Date"]
                series = series.drop("Date", axis=1)
            if not is_mitigated and parts[-1] == "deviated":
                series_set[DeviationSource[parts[1]]][DeviationScale[parts[2]]][SeriesColumn[parts[3]]] = series
            elif is_mitigated and parts[-1] == "mitigated":
                mitigation = series_set[DeviationSource[parts[1]]][DeviationScale[parts[2]]][SeriesColumn[parts[3]]] = {}
                mitigation[Mitigation.DATA] = series
                mitigation[Mitigation.TIME] = float(parts[4])

        return series_set
