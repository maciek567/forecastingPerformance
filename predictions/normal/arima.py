from numpy import ndarray
from pandas import DataFrame
from pandas import Series
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA
from statsmodels.tsa.arima.model import ARIMA

from predictions.normal import utils
from predictions.normal.prediction import Prediction, PredictionResults
from predictions.normal.utils import PredictionMethod
from timeseries.enums import SeriesColumn, DeviationSource


class ArimaPrediction(Prediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, mitigation_time: int = 0):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, mitigation_time)

    @staticmethod
    def print_elapsed_time(elapsed_time: float):
        print(f"Execution time: {elapsed_time} [ms]")

    def plot_extrapolation(self, prediction):
        utils.plot_extrapolation(self, prediction, PredictionMethod.Arima)


class ManualArima(ArimaPrediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, mitigation_time: int = 0):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, mitigation_time)

    def extrapolate_and_measure(self, params: dict) -> PredictionResults:
        return super().execute_and_measure(self.extrapolate, params)

    def find_d(self):
        return ndiffs(self.data_to_learn, test='adf')

    def extrapolate(self, params: dict) -> Series:
        data_with_prediction = self.data_to_learn.copy()
        for date, r in self.data_to_learn_and_validate.iloc[self.training_set_end:].items():
            model = ARIMA(data_with_prediction,
                          order=(params.get("p", 1), self.find_d(), params.get("q", 1))).fit()

            single_prediction = model.forecast()
            prediction_series = Series(single_prediction.values, index=[date])
            data_with_prediction = data_with_prediction.append(prediction_series)

        extrapolation = data_with_prediction[self.training_set_end:]
        return extrapolation


class AutoArimaPMD(ArimaPrediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, mitigation_time: int = 0):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, mitigation_time)
        self.auto_arima_model = None

    def extrapolate_and_measure(self, params: dict) -> PredictionResults:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> ndarray:
        self.auto_arima_model = auto_arima(self.data_to_learn,
                                           stat_p=params.get("p", 1),
                                           start_q=params.get("q", 1),
                                           test="adf",
                                           trace=True)
        periods = self.data_size - self.training_set_end
        return self.auto_arima_model.predict(n_periods=periods).values

    def print_summary(self):
        print(self.auto_arima_model.summary())


class AutoArimaSF(ArimaPrediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, mitigation_time: int = 0):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, mitigation_time)
        self.auto_arima_model = None

    def extrapolate_and_measure(self, params: dict) -> PredictionResults:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> ndarray:
        series_id = [0 for i in range(0, self.training_set_end)]
        series = DataFrame({"ds": self.data_to_learn.keys(), "y": self.data_to_learn.values, "unique_id": series_id})

        sf = StatsForecast(
            models=[AutoARIMA()],
            freq='D',
        )
        extrapolation = sf.forecast(df=series, h=self.data_size - self.training_set_end)

        return extrapolation.values[:, 1]
