import time

import numpy as np
from pandas import Series
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.models import (AutoCES, GARCH)
from statsmodels.tsa.arima.model import ARIMA

from predictions.prediction import Prediction, PredictionResults, PredictionStats
from predictions.utils import prepare_sf_dataframe, extract_predictions
from timeseries.enums import DeviationSource, DeviationScale


class ManualArima(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: int,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark, weights=weights)

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    def find_d(self):
        return ndiffs(self.data_to_learn, test='adf')

    def extrapolate(self, params: dict) -> PredictionResults:
        data_with_prediction = self.data_to_learn.copy()

        start_time = time.perf_counter_ns()
        for i in range(0, self.predict_size):
            model = ARIMA(data_with_prediction,
                          order=(params.get("p", 1), self.find_d(), params.get("q", 1))).fit()

            single_prediction = model.forecast()
            prediction_series = Series(single_prediction.values, index=[i])
            data_with_prediction = data_with_prediction.append(prediction_series)
        prediction_time = time.perf_counter_ns()

        extrapolation = data_with_prediction[self.training_size:]
        return PredictionResults(results=extrapolation, start_time=start_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return ManualArima


class AutoArimaPMD(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: int,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark=spark, weights=weights)
        self.auto_arima_model = None

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> PredictionResults:
        start_time = time.perf_counter_ns()
        self.auto_arima_model = auto_arima(self.data_to_learn,
                                           stat_p=params.get("p", 1),
                                           start_q=params.get("q", 1),
                                           test="adf",
                                           trace=True)

        periods = self.predict_size
        result = self.auto_arima_model.predict(n_periods=periods).values
        prediction_time = time.perf_counter_ns()

        return PredictionResults(results=result, start_time=start_time, prediction_time=prediction_time)

    def print_summary(self):
        print(self.auto_arima_model.summary())

    @staticmethod
    def get_method():
        return AutoArimaPMD


class AutoArima(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: int,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark, weights=weights)
        self.auto_arima_model = None

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> PredictionResults:
        df = prepare_sf_dataframe(self.data_to_learn, self.training_size)

        start_time = time.perf_counter_ns()
        sf = StatsForecast(
            models=[AutoARIMA(seasonal=False, max_order=8, start_p=4, start_q=4)],
            freq='D',
        )
        sf.fit(df=df)
        fit_time = time.perf_counter_ns()

        extrapolation = sf.predict(h=self.predict_size)
        prediction_time = time.perf_counter_ns()

        params = sf.fitted_[0][0].model_['arma']
        results = extract_predictions(extrapolation, "AutoARIMA")
        return PredictionResults(results=results, parameters=params,
                                 start_time=start_time, model_time=fit_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return AutoArima


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
        df['log'] = np.log(df['log'].astype(float))
        returns = df[['unique_id', 'ds', 'log']]
        returns = returns.rename(columns={'log': 'y'})

        start_time = time.perf_counter_ns()
        models = [
            # ARCH(1),
            # ARCH(2),
            # GARCH(1,1),
            # GARCH(1, 2),
            # GARCH(2, 1),
            GARCH(2, 2)
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

        selected_method = "GARCH(2,2)"
        garch = forecasts[selected_method]
        garch = garch.reset_index()
        garch["exp"] = np.exp(garch[selected_method])
        results = extract_predictions(garch, "exp")
        for column in results.keys():
            # series_to_multiply = self.data_to_learn[column].values[-1] / results[column].values[0]
            series_to_multiply = self.data_to_learn[column].values[(self.training_size[column] - self.predict_size):]
            results[column] = results[column].multiply(series_to_multiply)
        return PredictionResults(results=results,
                                 start_time=start_time, model_time=fit_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return Garch
