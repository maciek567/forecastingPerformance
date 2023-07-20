import time

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS
from pandas import Series
from pyEsn.ESN import ESN

from predictions.prediction import Prediction, PredictionStats, PredictionResults, PredictionResSimple
from predictions.utils import prepare_sf_dataframe, extract_predictions, cut_extrapolation
from timeseries.enums import DeviationSource, DeviationScale


class Reservoir(Prediction):
    def __init__(self, prices: dict, real_prices: dict, prediction_border: int, prediction_delay: dict,
                 columns: list, deviation: DeviationSource, scale: DeviationScale, mitigation_time: dict = None,
                 spark=None, weights=None):
        super().__init__(prices, real_prices, prediction_border, prediction_delay, columns, deviation, scale,
                         mitigation_time, spark, weights=weights)

    def extrapolate_and_measure(self, params: dict) -> PredictionStats:
        return super().execute_and_measure(self.extrapolate, params)

    def extrapolate(self, params: dict) -> PredictionResSimple:
        n_reservoir = 500
        sparsity = 0.2
        rand_seed = 23
        spectral_radius = 1.2
        noise = .0005
        res_dict = {}

        for column in self.columns:
            start_time = time.perf_counter_ns()
            esn = ESN(n_inputs=1,
                      n_outputs=1,
                      n_reservoir=n_reservoir,
                      sparsity=sparsity,
                      random_state=rand_seed,
                      spectral_radius=spectral_radius,
                      noise=noise)
            esn.fit(np.ones(self.training_size[column]), self.data_to_learn[column].values)
            fit_time = time.perf_counter_ns()

            prediction = esn.predict(np.ones(self.predict_size))
            prediction_time = time.perf_counter_ns()
            result = Series([pred[0] for pred in prediction])

            res_dict[column] = PredictionResults(results=result,
                                                 start_time=start_time, model_time=fit_time,
                                                 prediction_time=prediction_time)
        results = {column: results.results for column, results in res_dict.items()}
        results = cut_extrapolation(results, self.prediction_delay, self.columns, self.data_to_validate)
        return PredictionResSimple(results=results,
                                   model_time=sum([results.model_time for results in res_dict.values()]),
                                   prediction_time=sum([results.prediction_time for results in res_dict.values()]))

    @staticmethod
    def get_method():
        return Reservoir


class NHits(Prediction):
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
        config = dict(max_steps=2, val_check_steps=1, input_size=12,
                      mlp_units=3 * [[8, 8]])
        nf = NeuralForecast(
            models=[AutoNHITS(h=self.predict_size, config=config, num_samples=1)],
            freq='D'
        )
        nf.fit(df=df)
        fit_time = time.perf_counter_ns()

        extrapolation = nf.predict()
        prediction_time = time.perf_counter_ns()

        results = extract_predictions(extrapolation, "AutoNHITS")
        results = cut_extrapolation(results, self.prediction_delay, self.columns, self.data_to_validate)
        return PredictionResults(results=results, start_time=start_time, model_time=fit_time,
                                 prediction_time=prediction_time)

    @staticmethod
    def get_method():
        return NHits
