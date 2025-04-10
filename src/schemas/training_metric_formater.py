import time
from typing import List, Dict
from src.schemas.schemas.metric_schema import MetricSchema
from src.models.training_metrics_calculator import TrainingMetricsCalculator


class TrainingMetricsFormatter:
    """
    Uses TrainingMetricsCalculator to produce a MetricSchema.
    """

    def __init__(self):
        self._calculator = TrainingMetricsCalculator()

    def format(
        self,
        predictions: List[Dict],
        targets: List[Dict],
    ) -> MetricSchema:
        metrics = self._calculator.calculate(predictions, targets)
        return MetricSchema(metrics=metrics, timestamp=time.time())
