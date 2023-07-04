import time

from pandas import Series
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA
from statsmodels.tsa.arima.model import ARIMA

from predictions.prediction import Prediction, PredictionResults, PredictionStats
from predictions.utils import prepare_sf_dataframe
from timeseries.enums import SeriesColumn, DeviationSource, DeviationScale


class ArimaPrediction(Prediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, scale: DeviationScale, mitigation_time: int = 0,
                 spark=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, scale,
                         mitigation_time, spark)

    @staticmethod
    def print_elapsed_time(elapsed_time: float):
        print(f"Execution time: {elapsed_time} [ms]")


class ManualArima(ArimaPrediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, scale: DeviationScale, mitigation_time: int = 0,
                 spark=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, scale,
                         mitigation_time, spark)

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


class AutoArimaPMD(ArimaPrediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, scale: DeviationScale, mitigation_time: int = 0,
                 spark=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, scale,
                         mitigation_time, spark)
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


class AutoArimaSF(ArimaPrediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, scale: DeviationScale, mitigation_time: int = 0,
                 spark=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, scale,
                         mitigation_time, spark)
        self.auto_arima_model = None

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> PredictionResults:
        series = prepare_sf_dataframe(self.data_to_learn, self.training_size)

        start_time = time.perf_counter_ns()
        sf = StatsForecast(
            models=[AutoARIMA(seasonal=False, max_order=8, start_p=4, start_q=4)],
            freq='D',
        )
        sf.fit(df=series)
        fit_time = time.perf_counter_ns()

        extrapolation = sf.predict(h=self.predict_size)
        prediction_time = time.perf_counter_ns()

        result = extrapolation.values[:, 1]
        params = sf.fitted_[0][0].model_['arma']
        return PredictionResults(results=result, parameters=params,
                                 start_time=start_time, model_time=fit_time, prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return AutoArimaSF
