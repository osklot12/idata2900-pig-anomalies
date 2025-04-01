from dataclasses import dataclass
from typing import Dict

from src.schemas.schemas.schema import Schema


@dataclass(frozen=True)
class MetricSchema(Schema):
    """
    A schema for metrics.

    Attributes:
        metrics (Dict[str, Metric]): dictionary (metric type, value) of metrics
    """
    metrics: Dict[str, float]