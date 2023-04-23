import numpy as np
import pandas as pd
from pandas import Series
from pykalman import KalmanFilter

from timeseries.utils import SeriesColumn, DeviationScale, DeviationSource, Deviation


class NoisedSeries:
    def __init__(self, model, all_noises_strength: dict, partially_noised_strength: dict):
        self.model = model
        self.all_noises_strength = self.set_all_noises_strength(all_noises_strength)
        self.partially_noised_strength = partially_noised_strength
        self.set_all_noised_series()
        self.set_partially_noised_series()
        self.set_mitigated_noises_series()

    @staticmethod
    def set_all_noises_strength(all_noises_strength: dict):
        return all_noises_strength if all_noises_strength is not None \
            else {DeviationScale.SLIGHTLY: 0.4, DeviationScale.MODERATELY: 1.0, DeviationScale.HIGHLY: 3.0}

    def set_all_noised_series(self):
        self.model.all_deviated_series[DeviationSource.NOISE] = \
            {strength: self.noise_all_series(self.all_noises_strength[strength]) for strength in DeviationScale}

    def set_partially_noised_series(self):
        if self.partially_noised_strength is not None:
            self.model.partially_deviated_series[DeviationSource.NOISE] = \
                self.noise_some_series_set(self.partially_noised_strength)

    def set_mitigated_noises_series(self):
        self.model.mitigated_deviations_series[DeviationSource.NOISE] = \
            {strength: {
                column: self.apply_kalman(self.model.all_deviated_series[DeviationSource.NOISE][strength][column])
                for column in SeriesColumn}
             for strength in DeviationScale}

    def noise_all_series(self, power: float) -> dict:
        return self.model.deviate_all_series(
            {column: Deviation(self.add_noise, power) for column in SeriesColumn})

    def noise_some_series_set(self, partially_noised_strength) -> dict:
        return {strength: self.noise_some_series(
            {column: strengths[strength] for column, strengths in partially_noised_strength.items()})
            for strength in DeviationScale}

    def noise_some_series(self, noises: dict) -> dict:
        return self.model.deviate_some_series(
            {column: Deviation(self.add_noise, power) for column, power in noises.items()})

    def add_noise(self, data: Series, power: float) -> Series:
        mean = 0
        std_dev = power
        noise = np.random.normal(mean, std_dev, self.model.time_series_end - self.model.time_series_start)
        return data + noise

    @staticmethod
    def apply_kalman(series: Series):
        observations = pd.DataFrame(series)
        initial_value_guess = observations.iloc[0]
        observation_covariance = np.diag([0.5]) ** 2

        transition_covariance = np.diag([0.2]) ** 2
        kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
        )
        smoothed_series = kf.smooth(observations)[0]
        return Series([value[0] for value in smoothed_series.tolist()])

