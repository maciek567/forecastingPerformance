import sys

sys.path.append('..')
from hpc.shared import company_name, column
from predictions.hpc.modelHPC import PredictionModelHPC

from predictions.hpc.mlHPC import GBTRegressorHPC

prediction_start = 260
iterations = 5

model = PredictionModelHPC(company_name, prediction_start, column, iterations=iterations)

gbt = model.configure_model(GBTRegressorHPC, optimize=False)

gbt.compute_statistics_set(save_to_file=True)
