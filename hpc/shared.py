import sys

sys.path.append('..')
from timeseries.enums import DeviationScale, SeriesColumn

company_name = "Intel"
column = SeriesColumn.CLOSE
obsolescence_scale = {DeviationScale.SLIGHTLY: 5, DeviationScale.MODERATELY: 15, DeviationScale.HIGHLY: 50}
