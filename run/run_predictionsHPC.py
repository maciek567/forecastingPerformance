import sys

sys.path.append('..')
from run.shared import company_name, column
from predictions.hpc.modelHPC import PredictionModelHPC

from predictions.hpc.arimaHPC import AutoArimaHPC

prediction_start = 260
iterations = 2

model = PredictionModelHPC(company_name, prediction_start, column, iterations=iterations)

gbt = model.configure_model(AutoArimaHPC, optimize=False)

gbt.compute_statistics_set(save_to_file=True)
