from datetime import date, datetime, timedelta

from pandas import Series

from timeseries.utils import SeriesColumn, DeviationScale, DeviationSource

DAYS_IN_YEAR = 365


class ObsolescenceSeries:
    def __init__(self, model, obsoleteness_scale: dict):
        self.model = model
        self.obsolescence_scale = self.set_all_obsolete_parts(obsoleteness_scale)
        self.set_all_obsolete_series()

    @staticmethod
    def set_all_obsolete_parts(obsoleteness_scale: dict):
        return obsoleteness_scale if obsoleteness_scale is not None \
            else {DeviationScale.SLIGHTLY: 5, DeviationScale.MODERATELY: 15, DeviationScale.HIGHLY: 50}

    def set_all_obsolete_series(self):
        self.model.all_deviated_series[DeviationSource.TIMELINESS] = \
            {strength: self.obsolete_all_series(self.obsolescence_scale[strength]) for strength in DeviationScale}

    def obsolete_all_series(self, obsoleteness_scale: int) -> dict:
        return {column: self.single_series_with_extra_days(column.value, obsoleteness_scale) for column in SeriesColumn}

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
