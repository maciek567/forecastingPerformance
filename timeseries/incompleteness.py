import numpy as np
from pandas import Series
from scipy import interpolate

from timeseries.enums import DeviationScale, DeviationSource, Deviation, SeriesColumn
from timeseries.utils import perform_mitigation


class IncompleteSeries:
    def __init__(self, model, all_incomplete_parts: dict, partially_incomplete_parts: dict):
        self.model = model
        self.all_incomplete_parts = self.set_all_incomplete_parts(all_incomplete_parts)
        self.partially_incomplete_parts = self.set_partly_incomplete_parts(partially_incomplete_parts)
        self.set_all_incomplete_series()
        self.set_partially_incomplete_series()
        self.set_mitigated_incompleteness_series()

    @staticmethod
    def set_all_incomplete_parts(all_incomplete_parts: dict):
        return all_incomplete_parts if all_incomplete_parts is not None \
            else {DeviationScale.SLIGHTLY: 0.05, DeviationScale.MODERATELY: 0.12, DeviationScale.HIGHLY: 0.3}

    @staticmethod
    def set_partly_incomplete_parts(partially_incomplete_parts):
        if partially_incomplete_parts is not None:
            return {column: partially_incomplete_parts[column] if column in partially_incomplete_parts.keys()
            else {scale: 0 for scale in DeviationScale} for column in SeriesColumn}

    def set_all_incomplete_series(self):
        self.model.all_deviated_series[DeviationSource.INCOMPLETENESS] = \
            {strength: self.nullify_all_series(self.all_incomplete_parts[strength]) for strength in DeviationScale}

    def set_partially_incomplete_series(self):
        if self.partially_incomplete_parts is not None:
            self.model.partially_deviated_series[DeviationSource.INCOMPLETENESS] = \
                {scale: self.nullify_some_series(
                    {column: incomplete[scale] for column, incomplete in self.partially_incomplete_parts.items()})
                    for scale in DeviationScale}

    def set_mitigated_incompleteness_series(self):
        self.model.mitigated_deviations_series[DeviationSource.INCOMPLETENESS] = \
            {strength:
                {column: perform_mitigation(
                    self.model.all_deviated_series[DeviationSource.INCOMPLETENESS][strength][column],
                    self.apply_interpolation)
                    for column in self.model.columns}
                for strength in DeviationScale}

    def nullify_all_series(self, incomplete_part: float) -> dict:
        return self.model.deviate_all_series(
            {column: Deviation(self.add_incompleteness, incomplete_part) for column in self.model.columns})

    def nullify_some_series(self, incomplete_parts: dict) -> dict:
        return self.model.deviate_some_series(
            {column: Deviation(self.add_incompleteness, incomplete_part) for column, incomplete_part in
             incomplete_parts.items()})

    def add_incompleteness(self, data: Series, incomplete_part: float) -> Series:
        incompleteness = np.random.choice([0, 1], self.model.data_size,
                                          p=[incomplete_part, 1.0 - incomplete_part])
        incomplete_data = []
        for i in range(self.model.data_size):
            if incompleteness[i] == 1:
                incomplete_data.append(data[i])
            else:
                incomplete_data.append(np.nan)
        return Series(incomplete_data, index=data.index.tolist())

    @staticmethod
    def apply_interpolation(series: Series) -> Series:
        series_values = Series(series.values)
        x = series_values.dropna().index
        y = series_values.dropna()
        cubic_interpolation = interpolate.interp1d(x, y, kind="cubic", fill_value="extrapolate")
        return Series(cubic_interpolation(series_values.index), index=series.index.tolist())
