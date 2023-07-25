import os

import numpy as np
from pandas import DataFrame, read_csv, concat

from inout.intermediate import IntermediateProvider
from inout.paths import stock_path, aggregation_path
from run.configuration import time_series_start, time_series_values

company_name_label = "name"
min_label = "min"
max_label = "max"
avg_label = "avg"
std_label = "std dev"
coefficient_of_variation = "cv"
df = DataFrame(columns=[company_name_label, min_label, max_label, avg_label, std_label])

for file_name in os.listdir(stock_path):
    if file_name.endswith(".csv"):
        company_name = file_name.split(".")[0]
        full_series = read_csv(os.path.join(stock_path, file_name))
        series_start = full_series.index[full_series['Date'] == time_series_start].values.tolist()[0]
        series_end = series_start + time_series_values
        series = full_series["close"][series_start: series_end]
        result = {
            company_name_label: company_name,
            min_label: np.min(series),
            max_label: np.max(series),
            avg_label: np.mean(series),
            std_label: np.std(series),
            coefficient_of_variation: np.std(series) / np.mean(series) * 100}
        df = concat([df, DataFrame([result])])

name = "stock_stats"
IntermediateProvider.save_as_tex(df, aggregation_path, name, precision=2)

print("DONE")
