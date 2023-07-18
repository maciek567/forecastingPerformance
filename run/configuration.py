from timeseries.enums import SeriesColumn, DeviationScale

# company_names = ['AMD', 'Accenture', 'Acer', 'Activision', 'Adobe', 'Akamai', 'Alibaba', 'Amazon', 'Apple', 'At&t',
#                  'Autodesk', 'Canon', 'Capgemini', 'Cisco', 'Ericsson', 'Facebook', 'Google', 'HP', 'IBM', 'Intel',
#                  'Mastercard', 'Microsoft', 'Motorola', 'Nokia', 'Nvidia', 'Oracle', 'Sony', 'Tmobile']
# methods = [AutoArima, Ces, Garch, XGBoost, Reservoir, NHits, AutoArimaSpark, CesSpark, XGBoostSpark]

company_names = ["Facebook"]
time_series_start = "2017-01-03"
time_series_values = 1515

weights = {SeriesColumn.OPEN: 0.2, SeriesColumn.CLOSE: 0.2, SeriesColumn.ADJ_CLOSE: 0.25,
           SeriesColumn.HIGH: 0.15, SeriesColumn.LOW: 0.15, SeriesColumn.VOLUME: 0.05}
all_noises_scale = {DeviationScale.SLIGHTLY: 0.7, DeviationScale.MODERATELY: 1.7, DeviationScale.HIGHLY: 4.0}
all_incompleteness_scale = {DeviationScale.SLIGHTLY: 0.05, DeviationScale.MODERATELY: 0.12, DeviationScale.HIGHLY: 0.3}
all_obsolete_scale = {DeviationScale.SLIGHTLY: 5, DeviationScale.MODERATELY: 15, DeviationScale.HIGHLY: 50}
partially_noised_scales = \
    {SeriesColumn.CLOSE: {DeviationScale.SLIGHTLY: 0.6, DeviationScale.MODERATELY: 2.0, DeviationScale.HIGHLY: 6.0},
     SeriesColumn.OPEN: {DeviationScale.SLIGHTLY: 0.4, DeviationScale.MODERATELY: 1.7, DeviationScale.HIGHLY: 5.2}}
partially_incomplete_scales = \
    {SeriesColumn.CLOSE: {DeviationScale.SLIGHTLY: 0.05, DeviationScale.MODERATELY: 0.12, DeviationScale.HIGHLY: 0.3},
     SeriesColumn.OPEN: {DeviationScale.SLIGHTLY: 0.03, DeviationScale.MODERATELY: 0.08, DeviationScale.HIGHLY: 0.18}}
partially_obsolete_scales = \
    {SeriesColumn.CLOSE: {DeviationScale.SLIGHTLY: 5, DeviationScale.MODERATELY: 20, DeviationScale.HIGHLY: 50},
     SeriesColumn.OPEN: {DeviationScale.SLIGHTLY: 3, DeviationScale.MODERATELY: 12, DeviationScale.HIGHLY: 30}}
