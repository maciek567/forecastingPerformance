from numpy import ndarray
from pyspark.sql import DataFrame
from pyspark.sql.functions import min, max


def normalize(series: DataFrame, original_series: DataFrame) -> DataFrame:
    min_val = original_series.select(min("Values")).collect()
    max_val = original_series.select(max("Values")).collect()
    df = (series - min_val) / (max_val - min_val)
    df.columns = ["y"]
    return df


def denormalize(series: ndarray, original_series: DataFrame) -> ndarray:
    return series * (original_series.max() - original_series.min()) + original_series.min()
