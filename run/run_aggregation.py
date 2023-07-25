from util.aggregation import AggregationType, PredictionAggregation, MetricAggregation

if __name__ == '__main__':
    aggregation_type = AggregationType.METRIC

    aggregation = None
    if aggregation_type == AggregationType.PREDICTION:
        aggregation = PredictionAggregation()
    else:
        aggregation = MetricAggregation()
    aggregation.aggregate_results()

print("DONE")
