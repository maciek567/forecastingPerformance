from datetime import date, datetime, timedelta

from pandas import Series

from timeseries.enums import SeriesColumn, DeviationScale, DeviationSource

DAYS_IN_YEAR = 365


class ObsolescenceSeries:
    def __init__(self, model, all_obsolete_scale: dict, partly_obsolete_scale: dict):
        self.model = model
        self.all_obsolescence_scale = self.set_all_obsolete_parts(all_obsolete_scale)
        self.partially_obsolete_scales = self.set_partly_obsolete_parts(partly_obsolete_scale)
        self.set_all_obsolete_series()
        self.set_partly_obsolete_series()

    @staticmethod
    def set_all_obsolete_parts(obsoleteness_scale: dict):
        return obsoleteness_scale if obsoleteness_scale is not None \
            else {DeviationScale.SLIGHTLY: 5, DeviationScale.MODERATELY: 15, DeviationScale.HIGHLY: 50}

    @staticmethod
    def set_partly_obsolete_parts(partly_obsolete_scale):
        if partly_obsolete_scale is not None:
            return {column: partly_obsolete_scale[column] if column in partly_obsolete_scale.keys()
            else {scale: 0 for scale in DeviationScale} for column in SeriesColumn}

    def set_all_obsolete_series(self):
        self.model.all_deviated_series[DeviationSource.TIMELINESS] = \
            {strength: self.obsolete_all_series(self.all_obsolescence_scale[strength]) for strength in DeviationScale}

    def set_partly_obsolete_series(self):
        if self.partially_obsolete_scales is not None:
            self.model.partially_deviated_series[DeviationSource.TIMELINESS] = \
                {strength: self.obsolete_some_series(
                    {column: strengths[strength] for column, strengths in self.partially_obsolete_scales.items()})
                    for strength in DeviationScale}

    def obsolete_all_series(self, obsoleteness_scale: int) -> dict:
        return {column: self.single_series_with_extra_days(column.value, obsoleteness_scale) for column in
                self.model.columns}

    def obsolete_some_series(self, obsoleteness_scale: dict) -> dict:
        return {column: self.model.real_series[column] if column not in self.partially_obsolete_scales.keys()
        else self.single_series_with_extra_days(column.value, obsoleteness_scale[column])
                for column in SeriesColumn}

    def single_series_with_extra_days(self, column_name: SeriesColumn, extra_days: int) -> Series:
        series = Series(list(self.model.data[column_name]), index=self.model.data["Date"])
        return series[self.model.time_series_start:self.model.time_series_end + extra_days]

    def get_ages(self, measurement_time: int = None) -> tuple:
        dates = self.model.real_series[SeriesColumn.OPEN].index.tolist()
        today = date.today() if measurement_time is None else self.to_date(dates[-1]) + timedelta(days=measurement_time)
        time_diffs = [str(measurement_time + len(dates) - i) for i in range(len(dates))]
        ages = [(today - self.to_date(dates[i])).days / DAYS_IN_YEAR for i in range(len(dates))]
        return time_diffs, ages

    @staticmethod
    def to_date(date_string: str) -> date:
        return datetime.strptime(date_string, '%Y-%m-%d').date()
