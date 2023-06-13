import sys

sys.path.append('..')
from hpc.shared import company_name, column
from predictions.hpc.modelHPC import PredictionModelHPC

from predictions.hpc.mlHPC import XGBoostHPC, ReservoirHPC

prediction_start = 260
iterations = 5

model = PredictionModelHPC(company_name, prediction_start, column, iterations=iterations)

xgboost = model.configure_model(ReservoirHPC, optimize=False)

xgboost.compute_statistics_set(save_to_file=True)
