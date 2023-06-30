import pandas as pd
from pandas import Series
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA

from predictions import utils
from predictions.prediction import Prediction, PredictionResults
from timeseries.enums import SeriesColumn, DeviationSource


class ArimaPrediction(Prediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, mitigation_time: int = 0, spark=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, mitigation_time,
                         spark=spark)

    @staticmethod
    def print_elapsed_time(elapsed_time: float):
        print(f"Execution time: {elapsed_time} [ms]")


class AutoArimaSpark(ArimaPrediction):
    def __init__(self, prices: Series, real_prices: Series, prediction_border: int, prediction_delay: int,
                 column: SeriesColumn, deviation: DeviationSource, mitigation_time: int = 0, spark=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, column, deviation, mitigation_time,
                         spark)

    def extrapolate_and_measure(self, params: dict) -> PredictionResults:
        return super().execute_and_measure(self.extrapolate, params)

    @staticmethod
    def create_model():
        return StatsForecast(
            models=[AutoARIMA()],
            freq='D',
        )

    def extrapolate(self, params: dict):
        series_id = [0 for i in range(0, self.training_set_end)]
        series = pd.DataFrame({"ds": self.data_to_learn.keys(), "y": self.data_to_learn.values, "unique_id": series_id})
        series['unique_id'] = series['unique_id'].astype(str)
        sdf = self.spark.createDataFrame(series)

        sf_params = self.create_model()
        sf_params.fit(df=series)
        params = sf_params.fitted_[0][0].model_['arma']

        sf = self.create_model()
        extrapolation = sf.forecast(df=sdf, h=self.data_size - self.training_set_end, level=[90])
        result = extrapolation.toPandas()["AutoARIMA"]

        return PredictionResults(results=result, parameters=params)

    def plot_extrapolation(self, prediction, company_name, to_predict, save_file: bool = False):
        utils.plot_extrapolation(self, prediction, AutoArimaSpark, company_name, to_predict, save_file)
