from util.aggregation import AggregationType, PredictionAggregation, MetricAggregation, ComparisonAggregation

if __name__ == '__main__':
    aggregation_type = AggregationType.PREDICTION

    aggregation = None
    if aggregation_type == AggregationType.PREDICTION:
        aggregation = PredictionAggregation()
    elif aggregation_type == AggregationType.METRIC:
        aggregation = MetricAggregation()
    elif aggregation_type == AggregationType.COMPARISON:
        aggregation = ComparisonAggregation()
    aggregation.aggregate_results()

print("DONE")
