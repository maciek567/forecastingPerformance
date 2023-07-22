import pandas as pd
from pandas import DataFrame

from timeseries.enums import SeriesColumn


def method_name(method) -> str:
    return str(method)[str(method).index(".") + 1: -2].split(".")[-1]


unique_ids_mapping = {SeriesColumn.OPEN: 0,
                      SeriesColumn.CLOSE: 1,
                      SeriesColumn.ADJ_CLOSE: 2,
                      SeriesColumn.HIGH: 3,
                      SeriesColumn.LOW: 4,
                      SeriesColumn.VOLUME: 5}


def get_column_by_id(column_id):
    return list(unique_ids_mapping.keys())[list(unique_ids_mapping.values()).index(column_id)]


def prepare_sf_dataframe(data_to_learn_dict, training_size) -> DataFrame:
    df = pd.DataFrame(columns=["ds", "y", "unique_id"])
    for column, series in data_to_learn_dict.items():
        series_id = [unique_ids_mapping[column] for i in range(0, training_size[column])]
        df = pd.concat([df, pd.DataFrame({"ds": series.keys(), "y": series.values, "unique_id": series_id})])
    return df


def prepare_spark_dataframe(df, spark):
    df['unique_id'] = df['unique_id'].astype(str)
    return spark.createDataFrame(df)


def extract_predictions(df, results_name) -> dict:
    df = df.reset_index()
    df["unique_id"] = df["unique_id"].astype(int)
    dfs = dict(tuple(df.groupby("unique_id")))
    return {get_column_by_id(column_id): results[results_name] for column_id, results in dfs.items()}


def cut_extrapolation(extrapolation, prediction_delay, columns, data_to_validate):
    return {
        column: extrapolation[column][prediction_delay[column]:prediction_delay[column] + len(data_to_validate[column])]
        for column in columns}


def normalized_columns_weights(columns, weights):
    used_weights = {weight: weights[weight] for weight in weights if weight in columns}
    return {column: weight / sum(used_weights.values()) for column, weight in used_weights.items()}
