import time

import pandas as pd
from mlforecast import MLForecast
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
from xgboost import XGBRegressor

from predictions.prediction import Prediction, PredictionStats, PredictionResults
from predictions.utils import prepare_sf_dataframe, extract_predictions, cut_extrapolation
from timeseries.enums import DeviationSource, DeviationScale


class XGBoost(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: dict,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark, weights=weights)

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> PredictionResults:
        df = prepare_sf_dataframe(self.data_to_learn, self.training_size)
        df["ds"] = pd.to_datetime(df["ds"])

        start_time = time.perf_counter_ns()
        models = [XGBRegressor(random_state=0, n_estimators=100)]
        model = MLForecast(models=models,
                           freq='D',
                           lags=[1, 7, 14],
                           lag_transforms={
                               1: [(rolling_mean, 7), (rolling_max, 7), (rolling_min, 7)],
                           },
                           date_features=['dayofweek', 'month'],
                           num_threads=6)
        model.fit(df, id_col='unique_id', time_col='ds', target_col='y', static_features=[])
        fit_time = time.perf_counter_ns()

        extrapolation = model.predict(horizon=self.predict_size)
        prediction_time = time.perf_counter_ns()

        results = extract_predictions(extrapolation, "XGBRegressor")
        results = cut_extrapolation(results, self.prediction_delay, self.columns, self.data_to_validate)
        return PredictionResults(results=results,
                                 start_time=start_time, model_time=fit_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return XGBoost
