import glob
import os

from pandas import Series, DataFrame

from inout.paths import deviations_csv_path


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

    @staticmethod
    def get_csv_path(name: str) -> str:
        return os.path.join(deviations_csv_path, name) + ".csv"

    @staticmethod
    def remove_current_files():
        files = glob.glob(deviations_csv_path + "*")
        for f in files:
            os.remove(f)
