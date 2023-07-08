import time

import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import (AutoCES, GARCH, ARCH)

from predictions.prediction import Prediction, PredictionStats, PredictionResults
from predictions.utils import prepare_sf_dataframe, extract_predictions
from timeseries.enums import DeviationSource, DeviationScale


class Ces(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: int,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark, weights=weights)

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> PredictionResults:
        df = prepare_sf_dataframe(self.data_to_learn, self.training_size)

        start_time = time.perf_counter_ns()
        sf = StatsForecast(
            models=[AutoCES()],
            freq='D',
        )
        sf.fit(df=df)
        fit_time = time.perf_counter_ns()

        extrapolation = sf.predict(h=self.predict_size)
        prediction_time = time.perf_counter_ns()

        result = extract_predictions(extrapolation, "CES")
        return PredictionResults(results=result,
                                 start_time=start_time, model_time=fit_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return Ces


class Garch(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: int,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark, weights=weights)

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> PredictionResults:
        df = prepare_sf_dataframe(self.data_to_learn, self.training_size)
        df['log'] = df['y'].div(df.groupby('unique_id')['y'].shift(1))
        df['log'] = np.log(df['log'])
        returns = df[['unique_id', 'ds', 'log']]
        returns = returns.rename(columns={'log': 'y'})

        start_time = time.perf_counter_ns()
        models = [
                  # ARCH(1),
                  # ARCH(2),
                  GARCH(1, 1),
                  # GARCH(1, 2),
                  # GARCH(2, 2),
                  # GARCH(2, 1)
                  ]
        sf = StatsForecast(
            df=returns,
            models=models,
            freq='D',
            n_jobs=-1
        )
        sf.fit()
        fit_time = time.perf_counter_ns()

        forecasts = sf.predict(h=self.predict_size)
        prediction_time = time.perf_counter_ns()

        selected_method = "GARCH(1,1)"
        garch = forecasts[selected_method]
        garch = garch.reset_index()
        garch["exp"] = np.exp(garch[selected_method])
        garch["final"] = garch["exp"].multiply(df["y"][self.training_size-len(self.data_to_validate):].reset_index()["y"])

        return PredictionResults(results=garch["final"],
                                 start_time=start_time, model_time=fit_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return Garch
