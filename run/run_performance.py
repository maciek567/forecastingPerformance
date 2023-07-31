import sys
from predictions.conditions import get_method_by_name
from predictions.model import PredictionModel
from run.configuration import create_stock, columns
from timeseries.enums import DeviationSource, DeviationScale, DeviationRange

prediction_start = 1500
iterations = 2
method = get_method_by_name(str(sys.argv[sys.argv.index("--method") + 1]))
company_name = str(sys.argv[sys.argv.index("--company") + 1])
deviation_range = DeviationRange.ALL

sources = [DeviationSource.NOISE, DeviationSource.INCOMPLETENESS, DeviationSource.TIMELINESS]
scales = [DeviationScale.SLIGHTLY, DeviationScale.MODERATELY, DeviationScale.HIGHLY]
is_mitigation = True
graph_start = 1400
unique_ids = "--unique_ids" in sys.argv


stock = create_stock(company_name)
base_model = PredictionModel(stock, prediction_start, columns, graph_start, iterations=iterations,
                             deviation_sources=sources, deviation_scale=scales,
                             is_deviation_mitigation=is_mitigation, deviation_range=deviation_range,
                             unique_ids=unique_ids, is_save_predictions=False)
model = base_model.configure_model(method)
model.compute_statistics_set(save_file=True)

print("DONE")
