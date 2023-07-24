from pandas import Series

from inout.intermediate import IntermediateProvider
from inout.paths import deviations_csv_path
from inout.provider import YFinanceProvider
from timeseries.enums import SeriesColumn, DeviationSource, DeviationRange, MitigationType, DeviationScale
from timeseries.incompleteness import IncompleteSeries
from timeseries.noise import NoisedSeries
from timeseries.obsolescence import ObsolescenceSeries


class StockMarketSeries:
    def __init__(self, company_name: str, time_series_start: str, time_series_values: int, weights: dict,
                 all_noised_scale: dict = None, all_incomplete_scale: dict = None, all_obsolete_scale: dict = None,
                 partly_noised_scale: dict = None, partly_incomplete_scale: dict = None,
                 partly_obsolete_scale: dict = None,
                 columns: set = SeriesColumn, cache: bool = False):
        self.company_name = company_name
        self.data = YFinanceProvider.load_csv(company_name)
        self.time_series_start = self.find_start_date(time_series_start)
        self.time_series_end = self.time_series_start + time_series_values
        self.data_size = self.time_series_end - self.time_series_start
        self.columns = columns
        self.provider = IntermediateProvider()
        self.real_series = self.create_multiple_series()
        self.weights = self.normalize_weights(weights)
        self.all_deviated_series = {}
        self.partially_deviated_series = {}
        self.mitigated_deviations_series = {}
        self.noises = NoisedSeries(self, all_noised_scale, partly_noised_scale)
        self.incompleteness = IncompleteSeries(self, all_incomplete_scale, partly_incomplete_scale)
        self.obsolescence = ObsolescenceSeries(self, all_obsolete_scale, partly_obsolete_scale)
        if cache:
            self.cache_series_set()

    def find_start_date(self, time_series_start: str) -> int:
        return self.data.index[self.data['Date'] == time_series_start].values.tolist()[0]

    def create_multiple_series(self) -> dict:
        return {column: self.create_single_series(column.value) for column in self.columns}

    def create_single_series(self, column_name: SeriesColumn) -> Series:
        series = Series(list(self.data[column_name]), index=self.data["Date"])
        return series[self.time_series_start:self.time_series_end]

    @staticmethod
    def get_list_for_tuple(series: dict, i: int, columns: list = None) -> list:
        return [series[column][i] for column in (SeriesColumn if columns is None else columns)]

    @staticmethod
    def get_dict_for_tuple(series: dict, i: int, columns: list = None) -> dict:
        return {column: series[column][i] for column in (SeriesColumn if columns is None else columns)}

    def get_deviated_series(self, source: DeviationSource,
                            deviation_range: DeviationRange = DeviationRange.ALL) -> dict:
        return self.all_deviated_series[source] if deviation_range == DeviationRange.ALL \
            else self.partially_deviated_series[source]

    def get_mitigated_series(self, source: DeviationSource, deviation_range: DeviationRange) -> dict:
        if deviation_range == DeviationRange.ALL:
            return {scale: {column: self.mitigated_deviations_series[source][scale][column][MitigationType.DATA]
                            for column in SeriesColumn} for scale in DeviationScale}
        else:
            return {scale: {column: self.mitigated_deviations_series[source][scale][column][MitigationType.DATA] if
            self.noises.partially_noised_strength[column][scale] != 0 else self.real_series[column]
                            for column in SeriesColumn} for scale in DeviationScale}

    def deviate_all_series(self, deviations: dict) -> dict:
        return {column: deviation.method(self.real_series[column], deviation.scale) for column, deviation in
                deviations.items()}

    def deviate_some_series(self, series_to_deviate: dict) -> dict:
        return {column: self.real_series[column] if column not in series_to_deviate.keys() else None
                for column in SeriesColumn} | \
            {column: deviation.method(self.real_series[column], deviation.scale)
             for column, deviation in series_to_deviate.items()}

    @staticmethod
    def normalize_weights(weights):
        return {column: weight / sum([w for w in weights.values()]) for column, weight in weights.items()}

    def cache_series_set(self):
       # self.provider.remove_current_files()
        self.cache_real_series()
        self.cache_processed_series(self.all_deviated_series, False, "_deviated")
        self.cache_processed_series(self.mitigated_deviations_series, True, "_mitigated")

    def cache_real_series(self):
        for attribute, series in self.real_series.items():
            self.provider.save_as_csv(series, deviations_csv_path,
                                      f"{self.company_name}_{attribute.value.upper()}_real")

    def cache_processed_series(self, dictionary: dict, is_mitigation: bool, suffix: str):
        for deviation, scales in dictionary.items():
            for scale, attributes in scales.items():
                for attribute, series in attributes.items():
                    mitigation_time = ""
                    if is_mitigation:
                        mitigation_time = "_" + str(series[MitigationType.TIME])
                        series = series[MitigationType.DATA]
                    self.provider.save_as_csv(series, deviations_csv_path,
                                              f"{self.company_name}_{deviation.name}_{scale.name}_{attribute.name}{mitigation_time}{suffix}")

    def determine_real_and_deviated_columns(self, deviation_range, source, columns) -> tuple:
        deviated_columns = None
        if deviation_range == DeviationRange.ALL and source != DeviationSource.NONE:
            return [], columns
        else:
            if source == DeviationSource.NOISE:
                deviated_columns = self.deviated_columns(self.noises.partially_noised_strength)
            elif source == DeviationSource.INCOMPLETENESS:
                deviated_columns = self.deviated_columns(self.incompleteness.partially_incomplete_parts)
            elif source == DeviationSource.TIMELINESS:
                deviated_columns = self.deviated_columns(self.obsolescence.partially_obsolete_scales)
            else:
                return columns, []
            deviated_columns = [column for column in deviated_columns if column is not None and column in columns]
            real_columns = [column for column in columns if column not in deviated_columns]
            return real_columns, deviated_columns

    @staticmethod
    def deviated_columns(deviation_scale_dict):
        return [(column if not all(value == 0.0 for value in scale_dict.values()) else None) for column, scale_dict in
                deviation_scale_dict.items()]
