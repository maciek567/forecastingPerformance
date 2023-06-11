import sys

sys.path.append('..')
from hpc.shared import company_name, column
from predictions.modelHPC import PredictionModelHPC

from predictions.ml import XGBoost

prediction_start = 260
iterations = 5

model = PredictionModelHPC(company_name, prediction_start, column, iterations=iterations)

xgboost = model.configure_model(XGBoost, optimize=False)

xgboost.compute_statistics_set(save_to_file=True)
