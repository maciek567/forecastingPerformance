import numpy as np
from pandas import Series

from timeseries.utils import DeviationScale, SeriesColumn, DeviationSource, Deviation


class IncompleteSeries:
    def __init__(self, model, all_incomplete_parts: dict, partially_incomplete_parts: dict):
        self.model = model
        self.all_incomplete_parts = self.set_all_incomplete_parts(all_incomplete_parts)
        self.partially_incomplete_parts = partially_incomplete_parts
        self.set_all_incomplete_series()
        self.set_partially_incomplete_series()
        self.set_mitigated_incompleteness_series()

    @staticmethod
    def set_all_incomplete_parts(all_incomplete_parts: dict):
        return all_incomplete_parts if all_incomplete_parts is not None \
            else {DeviationScale.SLIGHTLY: 0.05, DeviationScale.MODERATELY: 0.12, DeviationScale.HIGHLY: 0.3}

    def set_all_incomplete_series(self):
        self.model.all_deviated_series[DeviationSource.INCOMPLETENESS] = \
            {strength: self.nullify_all_series(self.all_incomplete_parts[strength]) for strength in DeviationScale}

    def set_partially_incomplete_series(self):
        if self.partially_incomplete_parts is not None:
            self.model.partially_deviated_series[DeviationSource.INCOMPLETENESS] = \
                self.nullify_some_series_set(self.partially_incomplete_parts)

    def set_mitigated_incompleteness_series(self):
        self.model.mitigated_deviations_series[DeviationSource.INCOMPLETENESS] = \
            {strength: {column: self.apply_interpolation(
                self.model.all_deviated_series[DeviationSource.INCOMPLETENESS][strength][column])
                        for column in SeriesColumn}
             for strength in DeviationScale}

    def nullify_all_series(self, incomplete_part: float) -> dict:
        return self.model.deviate_all_series(
            {column: Deviation(self.add_incompleteness, incomplete_part) for column in SeriesColumn})

    def nullify_some_series_set(self, partially_incomplete_parts) -> dict:
        return {incomplete: self.nullify_some_series(
            {column: incompleted[incomplete] for column, incompleted in partially_incomplete_parts.items()})
            for incomplete in DeviationScale}

    def nullify_some_series(self, incomplete_parts: dict) -> dict:
        return self.model.deviate_some_series(
            {column: Deviation(self.add_incompleteness, incomplete_part) for column, incomplete_part in
             incomplete_parts.items()})

    def add_incompleteness(self, data: Series, incomplete_part: float) -> Series:
        incompleteness = np.random.choice([0, 1], self.model.time_series_end - self.model.time_series_start,
                                          p=[incomplete_part, 1.0 - incomplete_part])
        incomplete_data = []
        for i in range(0, self.model.time_series_end - self.model.time_series_start):
            if incompleteness[i] == 1:
                incomplete_data.append(data[i])
            else:
                incomplete_data.append(np.nan)
        return Series(incomplete_data)

    @staticmethod
    def apply_interpolation(series: Series):
        return series.interpolate()
